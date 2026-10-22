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
from tools.head_motion import raw_to_6dof, regress_head_motion
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
all_fif_files = [i for i in all_fif_files if "empty_room" not in i.stem]
all_fif_files = [i for i in all_fif_files if "ses_0" not in i._str]
all_ica_files = utils.get_files(proc_path, "*.fif", strings=["realtime", "ica.fif"], depth="all")
all_ica_files = [i for i in all_ica_files if "ses_0" not in i._str]
all_fif_files.sort()
all_ica_files.sort()
all_files = list(zip(all_fif_files, all_ica_files))

epochs_options = {
    "full_trial": ["dots_onset", -1.5, 4.6, 0],
    "dots": ["dots_onset", -0.25, 0.6, 0],
    "response": ["response_onset", -0.25, 3.5, -20],
}

# function that can also be accessed by importing this file
def epochs_maker(
        raw_path, ica_path, epoch_settings, trigger_mapping,
        filter=(None, None), hilbert=False, channel_list=None, trial_label="",
        head_motion_corr=True, decim=1, custom_events=None, save=True
):
    raw = read_raw_fif(raw_path, preload=True)
    
    if isinstance(channel_list, list):
        raw = raw.pick(channel_list)
    else:
        pass
    
    ica = read_ica(ica_path)
    raw = ica.apply(raw)

    if head_motion_corr:
        raw = raw_to_6dof(raw)

    if filter != (None, None):
        raw = raw.filter(picks="mag", *filter)
        if hilbert:
            raw = raw.apply_hilbert(picks="mag", envelope=True)

    trial_type, tmin, tmax, trigger_modifier = epoch_settings

    events_selection = {trigger_mapping[i]: i + trigger_modifier for i in trigger_mapping.keys() if trial_type in trigger_mapping[i]}
    if isinstance(custom_events, np.ndarray):
        events = custom_events
    else:
        events, event_ids = events_from_annotations(raw, event_id=events_selection)

    epochs = Epochs(
        raw, events, event_ids, tmin, tmax, decim=decim, baseline=None
    )
    if head_motion_corr:
        epochs = regress_head_motion(epochs)

    epoch_filename = f"{trial_label}_{raw_path.name}"
    epoch_filename = epoch_filename.replace("raw", "epo")
    epoch_path = raw_path.parent.joinpath(
        epoch_filename
    )
    if save:
        epochs.save(epoch_path, fmt="single", overwrite=True)
    else:
        return epochs


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

    epoch_type = "full_trial"
    trial_label = epoch_type.replace("_", "-")
    epochs_maker(
        fif_path, ica_path, epochs_options[epoch_type],
        trigger_mapping, trial_label=trial_label, 
        head_motion_corr=True, decim=3
    )

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)} Time elapsed: {time_elapsed} min"
    print(to_print)