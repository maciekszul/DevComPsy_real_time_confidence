import sys
import time
import utils
import numpy as np
import pandas as pd
from pathlib import Path
from mne import (
    set_log_level,
    read_epochs,
    concatenate_epochs,
)
from mne.io import (
    read_raw_fif
)
from mne.decoding import SSD
from mne.preprocessing import read_ica

set_log_level("CRITICAL")

# access to the paths and settings
settings = utils.load_json("settings.json")
trigger_mapping = utils.load_json("trigger_mapping.json")
trigger_mapping = {int(i): trigger_mapping[i] for i in trigger_mapping.keys()}
dataset_path = Path(settings["dataset_path"])
exclude_sub = settings["exclude_sub"]

raw_path = dataset_path.joinpath("MEG", "raw")
proc_path = dataset_path.joinpath("MEG", "processed")
beh_path = dataset_path.joinpath("BEH")

# specific file search and selection
fif_files = utils.get_files(proc_path, "*.fif", strings=["full-trial", "realtime", "epo.fif"])
raw_files = utils.get_files(proc_path, "*.fif", strings=["realtime", "raw.fif"])

# load basic data
subjects = utils.get_directories(proc_path, strings=["sub-"], depth="one")
subjects = [i.stem for i in subjects]
subjects = [i for i in subjects if not utils.many_is_in([f"sub-{j}" for j in exclude_sub], i)]

def concat_epoch_data(subject):
    subject_fifs = [i for i in fif_files if subject in i.stem]
    data = [read_epochs(i) for i in subject_fifs]
    channels = [data[0].info.ch_names[ix] for ix, typ in enumerate(data[0].info.get_channel_types()) if typ == "mag"]
    data = concatenate_epochs(data, add_offset=True, on_mismatch="ignore", verbose=False)
    data.crop(tmin=0.6, tmax=None)
    data = data.pick(channels)
    info = data.info
    data = data.get_data(picks=channels)
    return data, info, subject_fifs[0]

def raw_data_block(subject, block, resample=120):
    subject_raws = [i for i in raw_files if subject in i.stem]
    subject_raws.sort()
    raw_file = subject_raws[block]
    ica_file = raw_file.parent.joinpath(raw_file.name.replace("_raw.fif", "_ica.fif"))
    ica = read_ica(ica_file)
    raw = read_raw_fif(raw_file, preload=True)
    channels = [raw.info.ch_names[ix] for ix, typ in enumerate(raw.info.get_channel_types()) if typ == "mag"]
    raw = ica.apply(raw)
    raw = raw.pick(channels)
    raw = raw.resample(resample)
    info = raw.info
    raw = raw.get_data()
    return raw, info, raw_file


def do_the_SSD(data, info, path, freqs_dict, name="_epochs_"):
    for freq in freqs_dict.keys():
        output_pkl = path.parent.joinpath(f"SSD_{freq}_{subject}_{name}.pkl")
        target, noise = freqs2ssd[freq]
        decomp = SSD(
            info=info,
            n_components=10,
            filt_params_signal=dict(
                l_freq=target[0],
                h_freq=target[1],
                l_trans_bandwidth=1,
                h_trans_bandwidth=1,
            ),
            filt_params_noise=dict(
                l_freq=noise[0],
                h_freq=noise[1],
                l_trans_bandwidth=1,
                h_trans_bandwidth=1,
            ),
            return_filtered=True,
            rank="full"
        )
        decomp.fit(data)
        utils.save_pickle(decomp, output_pkl)


if __name__ == '__main__':
    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no subject index")
    
    start_time = time.time()

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(subjects)}"
    print(to_print)

    subject = subjects[index]

    # freqs2ssd = {
    #     "alpha": [[8, 12], [7,13]], # [target, noise] frequency
    #     # "beta": [[13, 30], [12, 31]]
    # }

    # for i in range(4):
    #     data, info, path = raw_data_block(subject, i, resample=120)
    #     block = f"block-{str(i).zfill(3)}"
    #     do_the_SSD(data, info, path, freqs2ssd, name=block)


    freqs2ssd = {
            # "alpha": [[8, 12], [7,13]], # [target, noise] frequency
            "beta": [[13, 30], [12, 31]]
        }

    data, info, path = concat_epoch_data(subject)
    do_the_SSD(data, info, path, freqs2ssd, name="epochs")

    status = f"END: {subject}"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(subjects)} Time elapsed: {time_elapsed} min"
    print(to_print)