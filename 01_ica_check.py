import sys
import utils
from pathlib import Path
from mne import set_log_level
from mne.io import (
    read_raw_fif
)
from mne.preprocessing import read_ica


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


# function that can also be accessed by importing this file
def ica_check(fif_path, ica_path):
    raw = read_raw_fif(fif_path, preload=True)
    raw = raw.crop(tmin=100, tmax=200).filter(1, 30)
    ica = read_ica(ica_path)

    ica.plot_components()
    ica.plot_sources(raw, block=True, theme="dark")

    print(fif_path.name, "excluded comps: ", ica.exclude)

    ica.save(ica_path, overwrite=True)


if __name__ == '__main__':
    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no file index")
    
    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)}"
    print(to_print)
    
    fif_path, ica_path = all_files[index]
    ica_check(fif_path, ica_path)