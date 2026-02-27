import sys
import time
import utils
import numpy as np
from copy import copy
from pathlib import Path
from mne import set_log_level
from meegkit.dss import dss_line_iter
from mne.io import (
    read_raw_ctf,
    RawArray
)
from mne.preprocessing import ICA
from mne import (
    find_events,
    annotations_from_events
)

set_log_level("ERROR")

# access to the paths and settings
settings = utils.load_json("settings.json")
trigger_mapping = utils.load_json("trigger_mapping.json")
trigger_mapping = {int(i): trigger_mapping[i] for i in trigger_mapping.keys()}
dataset_path = Path(settings["dataset_path"])
raw_path = dataset_path.joinpath("MEG", "raw")
proc_path = dataset_path.joinpath("MEG", "processed")
beh_path = dataset_path.joinpath("BEH")

# specific file search and selection
all_files = utils.get_directories(raw_path, depth="all")
all_files = [i for i in all_files if "realtime" in i.parts[-1]]

# function that can also be accessed by importing this file
def process_ds(ds, trigger_mapping, proc_path):

    subject = ds.stem.split("_")[0]
    subject_path = utils.make_directory(proc_path, f"sub-{subject}")

    if "01.ds" in ds.parts[-1]:
        block_type = "calibration"
    else:
        block = int(ds.parts[-1].split("_")[-1].split(".")[0]) - 2
        block_type = str(block).zfill(3)
    filename = f"realtime_sub-{subject}_block-{block_type}_raw.fif"
    raw_output = subject_path.joinpath(filename)
    filename = f"realtime_sub-{subject}_block-{block_type}_ica.fif"
    ica_output = subject_path.joinpath(filename)
    
    raw = read_raw_ctf(ds, preload=True, clean_names=True, verbose=False)

    if "01.ds" in ds.parts[-1]:
        gripper_channels = ["UADC009", "UADC010"]
        raw = raw.pick(picks=gripper_channels)
        meg_calibration = {}
        for ch_name in ["UADC009", "UADC010"]:
            data = raw[ch_name][0]
            meg_calibration[ch_name] = [np.median(data).astype(float), np.max(data).astype(float)]
        filename = f"realtime_sub-{subject}_meg-calibration.json"
        calib_output = subject_path.joinpath(filename)
        utils.save_dict_as_json(calib_output, meg_calibration)

    else:
        raw = raw.apply_gradient_compensation(3)
        raw = raw.filter(None, 125)

        set_ch = {
            "EEG057":"eog", 
            "EEG058": "eog",
            "UDIO001": "stim"
        }

        raw = raw.set_channel_types(set_ch)
        events = find_events(raw, "UDIO001")

        # zapline

        raw_data = raw.get_data()
        raw_info = raw.info
        first_samp = raw.first_samp
        mag_ix = np.array([i for i, lab in enumerate(raw.get_channel_types()) if lab == "mag"])

        rd = np.array_split(raw_data[mag_ix], 10, axis=1)
        rd = [np.moveaxis(i, [0,1], [1,0]) for i in rd]
        rd_f = []

        for ix, ss in enumerate(rd):
            print(ds.name, f"{ix+1}/10")
            data, iters = dss_line_iter(
                ss, fline=50.0, sfreq=raw_info["sfreq"], 
                spot_sz=5.5, win_sz=10, nfft=1024, n_iter_max=30
            )
            rd_f.append(data)
        del rd
        rd_f = [np.moveaxis(i, [0,1], [1,0]) for i in rd_f]
        rd_f = np.hstack(rd_f)

        new_raw_data = copy(raw_data)
        del raw_data
        new_raw_data[mag_ix, :] = rd_f

        new_raw = RawArray(
            new_raw_data,
            raw_info,
            first_samp=first_samp
        )

        del raw

        # zapline
        afe = annotations_from_events(events, new_raw.info["sfreq"], trigger_mapping, new_raw.first_samp)

        new_raw = new_raw.set_annotations(afe)
    
        new_raw.save(raw_output, fmt="single", overwrite=True, verbose=False)
    
        new_raw = new_raw.filter(1, 40)
        ica = ICA(n_components=20)
        ica.fit(new_raw)
        ica.save(ica_output, overwrite=True, verbose=False)




if __name__ == '__main__':
    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no file index")
    
    ds = all_files[index]
    
    start_time = time.time()

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)}"
    print(to_print)
    
    process_ds(ds, trigger_mapping, proc_path)

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)} Time elapsed: {time_elapsed} min"
    print(to_print)