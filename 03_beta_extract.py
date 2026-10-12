import sys
import time
import utils
import numpy as np
import pandas as pd
from specparam import SpectralModel
from tools.burst_detection import extract_bursts
from tools.signal import regress_signal
from pathlib import Path
from mne import (
    set_log_level,
    read_epochs,
)

set_log_level("ERROR")

# access to the paths and settings
settings = utils.load_json("settings.json")
trigger_mapping = utils.load_json("trigger_mapping.json")
trigger_mapping = {int(i): trigger_mapping[i] for i in trigger_mapping.keys()}
dataset_path = Path(settings["dataset_path"])
exclude_sub = settings["exclude_sub"]

raw_path = dataset_path.joinpath("MEG", "RAW")
proc_path = dataset_path.joinpath("MEG", "processed")
beh_path = dataset_path.joinpath("BEH")

# specific file search and selection
fif_files = utils.get_files(proc_path, "*.fif", strings=["full-trial", "realtime", "epo.fif"])

def bursts_power_beh(data, behaviour, channel_name, times, sfreq):
    
    tf_trials = np.array([utils.superlet_tf(i, sfreq, num=200) for i in data])
    
    freqs = np.linspace(1, 120, num=200)
    
    search_range = np.where((freqs >= 10) & (freqs <= 33))[0]
    
    beta_lims = [13, 30]

    pds_mean_trials = np.mean(tf_trials, axis=(2, 0))

    sm = SpectralModel(peak_width_limits=(1.65, 12))

    sm.fit(freqs, pds_mean_trials, freq_range=[1,120])

    aperiodic_spectrum = 10**(sm._ap_fit)[search_range].reshape(-1,1)

    tf_trials = tf_trials[:,search_range,:]

    beta_power = tf_trials.mean(axis=1)

    bursts = extract_bursts(
        data, tf_trials, times, search_range, beta_lims, aperiodic_spectrum, sfreq
    )

    output_dict = {}
    for key in bursts.keys():
        if key == "waveform_times":
            continue
        elif key != "waveform":
            output_dict[key] = bursts[key].tolist()

    output_dict["channel"] = np.repeat(channel_name, bursts["trial"].shape).tolist()
    output_dict["subject"] = np.repeat(subject, bursts["trial"].shape).tolist()

    waveforms = bursts["waveform"]

    total_output = pd.DataFrame.from_dict(output_dict)
    total_output["trial_number"] = total_output["trial"]
    total_output = total_output.merge(behaviour, on="trial_number")

    return total_output, beta_power, waveforms



if __name__ == '__main__':
    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no subject index")
    
    start_time = time.time()

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(fif_files)}"
    print(to_print)

    fif_file = fif_files[index]

    subject = fif_file.stem.split("_")[-3]
    fif_block = fif_file.stem.split("_")[-2]
    beh_block = fif_block.replace("block-", "block-main_")

    beh_file = utils.get_files(beh_path, "*.csv", strings=[subject, beh_block])[0]

    behaviour = pd.read_csv(beh_file, index_col=False)

    epochs = read_epochs(fif_file)

    beh_fif_equivalent = np.mean(behaviour.trial_trigger.to_numpy() == epochs.events[:,2])

    if beh_fif_equivalent != 1.0:
        print("Epochs and Behaviour are not matching")
        sys.exit()

    epochs = regress_signal(epochs, ch_names=["UADC009", "UADC010"])

    channels = [epochs.info.ch_names[ix] for ix, typ in enumerate(epochs.info.get_channel_types()) if typ == "mag"]

    times = epochs.times
    sfreq = epochs.info["sfreq"]

    output_path = utils.make_directory(fif_file.parent, "burst_data")

    for ch_ix, channel_name in enumerate(channels):
        data = epochs.get_data(picks=channel_name)[:, 0, :]
        bursts_info, beta_power, waveforms = bursts_power_beh(
            data, behaviour, channel_name, times, sfreq
        )

        filename_core = f"_realtime_{channel_name}_{subject}_{fif_block}"

        path_burst_info = output_path.joinpath(
            f"bursts-beh-info{filename_core}.csv"
        )
        bursts_info.to_csv(path_burst_info, index_label=False)

        path_burst_wave = output_path.joinpath(
            f"bursts-waveforms{filename_core}.npy"
        )
        np.save(path_burst_wave, waveforms, allow_pickle=False)

        path_beta_power = output_path.joinpath(
            f"beta_power{filename_core}.npy"
        )
        np.save(path_beta_power, beta_power, allow_pickle=False)

        status = "CHANNEL PROCESSED"
        time_elapsed =  np.round((time.time() - start_time)/60, 2)
        to_print = f"{status} File:{str(index+1).zfill(3)}/{len(fif_files)} Channel: {str(ch_ix+1).zfill(3)}/{len(channels)}"
        print(to_print)

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(fif_files)} Time elapsed: {time_elapsed} min"
    print(to_print)


