import sys
import time
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from specparam import SpectralModel
from tools.burst_detection import extract_bursts
from pathlib import Path
from mne import (
    set_log_level,
    read_epochs,
    concatenate_epochs
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

# load basic data
subjects = utils.get_directories(proc_path, strings=["sub-"], depth="one")
subjects = [i.stem for i in subjects]


if __name__ == '__main__':
    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no subject index")
    
    start_time = time.time()

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(subjects)}"
    print(to_print)

    # load basic data
    subjects = utils.get_directories(proc_path, strings=["sub-"], depth="one")
    subjects = [i.stem for i in subjects]

    subject = subjects[index]

    beh_files = utils.get_files(beh_path, "*.csv", strings=["main_", subject])
    fif_files = utils.get_files(proc_path, "*.fif", strings=["dots-onset", "realtime", "_long-epoch_", subject, "epo.fif"])
    behaviour = pd.concat([pd.read_csv(i) for i in beh_files]).reset_index(drop=True)
    epochs = concatenate_epochs([read_epochs(i, verbose=False).apply_baseline((-0.25, 0.0)) for i in fif_files], add_offset=True, on_mismatch="ignore", verbose=False)
    overlap = np.mean(epochs.events[:,2] == behaviour.trial_trigger.to_numpy()) * 100
    print(f"epoch order overlaps in {overlap}%")

    # select sensors
    sens = ["C1", "C25", "C32", "C42", "C54", "C55", "C63"]
    CL = [i for i in epochs.ch_names if utils.many_is_in(["MLC"], i)]
    CL = [i for i in CL if not utils.many_is_in(sens, i)]
    CR = [i for i in epochs.ch_names if utils.many_is_in(["MRC"], i)]
    CR = [i for i in CR if not utils.many_is_in(sens, i)]
    chan_f = [31, 32, 33, 34, 41, 42, 43, 44, 51, 52, 53, 54]
    sens_F = [f"LF{i}" for i in chan_f]
    FL = [i for i in epochs.ch_names if utils.many_is_in(sens_F, i)]
    sens_F = [f"RF{i}" for i in chan_f]
    FR = [i for i in epochs.ch_names if utils.many_is_in(sens_F, i)]

    channels_used = CR + CL + FL + FR
    channels_used.sort()

    dict_list = []
    wfvm_list = []

    # output paths
    epoch_type, filter, experiment, subject, block, ftype = fif_files[0].stem.split("_")
    filename = "_".join(["burst-waveforms", epoch_type, experiment, subject + ".npy"])
    waveform_path = fif_files[0].parent.joinpath(filename)
    filename = "_".join(["burst-features", epoch_type, experiment, subject + ".npy"])
    features_path = fif_files[0].parent.joinpath(filename)

    # channel iteration
    for ix, channel in enumerate(channels_used):
        data = epochs.get_data(picks=channels_used[0])[:3,0,:]
        tf_trials = np.array([utils.superlet_tf(i, epochs.info["sfreq"]) for i in data])

        freqs = np.linspace(1, 120, num=tf_trials.shape[1])
        search_range = np.where((freqs >= 10) & (freqs <= 33))[0]
        beta_lims = [13, 30]

        pds_mean_trials = np.mean(tf_trials, axis=(2, 0))

        sm = SpectralModel(peak_width_limits=(1.65, 12))

        sm.fit(freqs, pds_mean_trials, freq_range=[1,120])

        aperiodic_spectrum = 10**(sm._ap_fit)[search_range].reshape(-1,1)

        tf_trials = tf_trials[:,search_range,:]

        bursts = extract_bursts(data, tf_trials, epochs.times, search_range, beta_lims, aperiodic_spectrum, epochs.info["sfreq"])

        output_dict = {}

        for key in bursts.keys():
            if key == "waveform_times":
                continue
            elif key != "waveform":
                output_dict[key] = bursts[key].tolist()

        output_dict["channel"] = np.repeat(channel, bursts["trial"].shape).tolist()
        output_dict["subject"] = np.repeat(subject, bursts["trial"].shape).tolist()
        dict_list.append(output_dict)
        wfvm_list.append(bursts["waveform"])
        print(f"{subject} channel: {ix+1}/{len(channels_used)}")

    output_dict = pd.concat([pd.DataFrame.from_dict(i) for i in dict_list])
    waveforms = np.vstack(wfvm_list)

    output_dict.to_csv(features_path, index=False)
    np.save(waveform_path, waveforms)

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(subjects)} Time elapsed: {time_elapsed} min"
    print(to_print)