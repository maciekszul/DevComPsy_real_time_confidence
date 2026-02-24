import sys
import time
import utils
import numpy as np
from pathlib import Path
from mne import (
    set_log_level,
    events_from_annotations,
    Epochs
)
from mne.io import (
    read_raw_fif
)
from mne.preprocessing import read_ica

# TO DO:
#  - integrate head motion regression



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
all_fif_files = utils.get_files(proc_path, "*.fif", strings=["realtime", "raw.fif"], depth="all")
all_fif_files = [i for i in all_fif_files if "calibration" not in i.stem]
all_ica_files = utils.get_files(proc_path, "*.fif", strings=["realtime", "ica.fif"], depth="all")
all_fif_files.sort()
all_ica_files.sort()
all_files = list(zip(all_fif_files, all_ica_files))

epochs_options = {
    "whole_trial": ["dots_onset", -1.5, 4.1, 0],
    "dots": ["dots_onset", -0.25, 0.6, 0],
    "response": ["response_onset", -0.25, 3.5, -20],
}

# function that can also be accessed by importing this file
def epochs_maker(fif_path, ica_path, epoch_settings, trigger_mapping, filter=(None, None), decim=0, limited_channels=True):
    """
    Parameters
    ----------

    epoch setting: list
        contains string to select annotated triggers, tmin, tmax
    
    """

    raw = read_raw_fif(fif_path, preload=True)
    filter_name = "nf"
    if filter != (None, None):
        filter_name = f"{filter[0]}-{filter[1]}"
        raw = raw.filter(*filter)
    ica = read_ica(ica_path)

    raw = ica.apply(raw)

    if limited_channels:
        mags = [ix for ix, ch_t in enumerate(raw.get_channel_types()) if ch_t == "mag"]
        grippers = [ix for ix, ch_n in enumerate(raw.ch_names) if utils.check_many(["UADC009", "UADC010"], ch_n, func="any")]
        channels = mags + grippers
        channels.sort()
        channels = [raw.ch_names[i] for i in channels]
        raw = raw.pick(channels)
    
    trial_type, tmin, tmax, modifier = epoch_settings

    events_selection = {trigger_mapping[i]: i + modifier for i in trigger_mapping.keys() if trial_type in trigger_mapping[i]}

    events, event_ids = events_from_annotations(raw, event_id=events_selection)

    epochs = Epochs(
        raw, events, event_ids, tmin, tmax, decim=decim,
    )

    trial_label = trial_type.replace("_", "-")
    filename = "_".join([trial_label] + [filter_name] + (fif_path.stem.split("_")[:-1]) + ["epo.fif"])
    epoch_path = Path(fif_path.parent).joinpath(filename)
    print(epoch_path)
    # epochs.save(epoch_path, fmt="single", overwrite=True)


if __name__ == '__main__':
    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no file index")
    
    start_time = time.time()

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)}"
    print(to_print)

    fif_path, ica_path = all_files[index]
    epochs_maker(
        fif_path, ica_path, epochs_options["whole_trial"], trigger_mapping,
        filter=(None, None), decim=2, limited_channels=True
    )

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)} Time elapsed: {time_elapsed} min"
    print(to_print)