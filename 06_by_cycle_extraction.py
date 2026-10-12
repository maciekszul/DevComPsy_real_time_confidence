import time
import sys
import emd
import utils
import pandas as pd
import numpy as np
from pathlib import Path
from mne import (
    concatenate_epochs,
    read_epochs,
    EpochsArray,
    create_info,
    set_log_level
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


set_log_level("CRITICAL")

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


def prepare_data(subject, decimate=2, baseline=False):
    beh_files = utils.get_files(beh_path, "*.csv", strings=["main_", subject])
    beh_files = [i for i in beh_files if "ses_0" not in i._str]
    fif_files = utils.get_files(proc_path, "*.fif", strings=["dots-onset", "realtime", "epo.fif", subject])
    fif_files = [i for i in fif_files if "ses_0" not in i._str]
    meg_calib_file = utils.get_files(proc_path, "*.json", strings=["meg-calibration", subject])[0]
    ssd_file = utils.get_files(proc_path, "*.pkl", strings=[subject])[0]
    ssd = utils.load_pickle(ssd_file)

    behaviour = pd.concat([pd.read_csv(i) for i in beh_files]).reset_index(drop=True)
    behaviour = behaviour[[i for i in behaviour.columns.tolist() if "exp_timing" not in i]]
    behaviour = behaviour.reset_index(names=["trial"])
    conditions = behaviour.groupby(["counterfactual_label", "coherence_level"]).rt.count().reset_index()
    conditions = conditions[["counterfactual_label", "coherence_level"]]
    behaviour = behaviour.assign(confidence_rank=0.0)
    for ix, (cf, ch) in conditions.iterrows():
        selection_map = (
            (behaviour.counterfactual_label == cf) &
            (behaviour.coherence_level == ch)
        )
        ranked_confidence = behaviour[selection_map].scale_response.rank(pct=True, na_option="bottom").to_numpy()
        behaviour.loc[selection_map, "confidence_rank"] = ranked_confidence
    epochs = concatenate_epochs([read_epochs(i, verbose=False) for i in fif_files], add_offset=True, on_mismatch="ignore", verbose=False)
    if baseline:
        epochs = epochs.apply_baseline(baseline)
    overlap = np.mean(epochs.events[:,2] == behaviour.trial_trigger.to_numpy()) * 100
    print(f"epoch order overlaps in {overlap}%")

    info = epochs.info
    times = epochs.times

    alpha_components = ssd.transform(epochs.get_data(picks="mag"))

    return info, times, epochs, alpha_components, meg_calib_file.parent


def trial_by_cycle(epoch_data, alpha_comp, info, times, epoch_ix, max_imfs=1, imf_ix=0, phase_size=36):
    imf = emd.sift.sift(alpha_comp, max_imfs=max_imfs)
    IP, IF, IA = emd.spectra.frequency_transform(imf[:,imf_ix], info["sfreq"]/2, 'nht')
    IP, IF, IA = IP.flatten(), IF.flatten(), IA.flatten()
    mask = minmax_scale(IA) > 0.05
    cycles = emd.cycles.get_cycle_vector(IP, return_good=True, mask=mask)
    cycles = cycles.flatten()

    df_data = {}

    pa_time = emd.cycles.phase_align(
        IP,
        times,
        cycles=cycles
    )[0]

    df_data["cycle_time_mean"] = pa_time.mean(axis=0)
    df_data["cycle_duration"] = pa_time[-1] - pa_time[0]
    df_data["cycle_on"] = pa_time[0]
    df_data["cycle_off"] = pa_time[-1]
    df_data["epoch_index"] = np.tile(epoch_ix, df_data["cycle_time_mean"].shape[0])
    df_data["cycle_index"] = np.arange(df_data["cycle_time_mean"].shape[0])

    pa_sensors = np.array([emd.cycles.phase_align(IP, sensor, cycles=cycles, npoints=phase_size)[0].T for sensor in epoch_data])
    pa_imf = emd.cycles.phase_align(IP, imf[:,imf_ix], cycles=cycles, npoints=phase_size)[0].T
    pa_amp = emd.cycles.phase_align(IP, IA, cycles=cycles, npoints=phase_size)[0].T
    pa_freq = emd.cycles.phase_align(IP, IF, cycles=cycles, npoints=phase_size)[0].T

    pa_mode_peak_ix = np.argmax(pa_imf, axis=1)
    pa_mode_trough_ix = np.argmin(pa_imf, axis=1)
    df_data["pa_mode_peak_ix"] = pa_mode_peak_ix
    df_data["pa_mode_trough_ix"] = pa_mode_trough_ix
    df_data["pa_mode_peak_amp"] = imf[pa_mode_peak_ix, imf_ix]
    df_data["pa_mode_trough_amp"] = imf[pa_mode_trough_ix, imf_ix]
    pa_freq_peak_max, pa_freq_trough_max = [np.max(i, axis=1) for i in np.split(pa_freq, 2, axis=1)]
    pa_amp_peak_max, pa_amp_trough_max = [np.max(i, axis=1) for i in np.split(pa_amp, 2, axis=1)]
    df_data["pa_freq_peak_max"] = pa_freq_peak_max
    df_data["pa_freq_trough_max"] = pa_freq_trough_max
    df_data["pa_amp_peak_max"] = pa_amp_peak_max
    df_data["pa_amp_trough_max"] = pa_amp_trough_max

    by_cycle_data = np.vstack([pa_sensors, pa_imf[np.newaxis, :], pa_amp[np.newaxis, :], pa_freq[np.newaxis, :]])
    by_cycle_data = np.moveaxis(by_cycle_data, [0, 1, 2], [1, 0, 2])
    
    df_data = pd.DataFrame.from_dict(df_data)

    return df_data, by_cycle_data



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

    info, times, epochs, alpha_components, subject_path = prepare_data(
        subject, decimate=1, baseline=(-0.25, 0.0)
    )

    by_cycle_info = create_info(
        info.ch_names + ["IMF", "pa_amplitude", "pa_frequency"],
        info["sfreq"],
        info.get_channel_types() + ["misc", "misc", "misc"]
    )

    imf_ix = 0
    dfs = []
    cyc_data = []
    for epoch_ix in range(len(epochs)):
        epoch = epochs[epoch_ix].get_data()[0]
        alpha_comp = alpha_components[epoch_ix][imf_ix]
        df, cyc_sens = trial_by_cycle(
            epoch, alpha_comp, info, times, epoch_ix
        )
        dfs.append(df)
        cyc_data.append(cyc_sens)
    
    dfs = pd.concat(dfs).reset_index(drop=True)
    cyc_data = np.vstack(cyc_data)

    cyc_data = EpochsArray(
        cyc_data,
        by_cycle_info
    )

    fif_output_filepath = subject_path.joinpath(f"cycle-by-cycle_{subject}.fif")
    csv_output_filepath = subject_path.joinpath(f"cycle-by-cycle_{subject}.csv")

    cyc_data.save(fif_output_filepath, fmt="single", overwrite=True)

    dfs.to_csv(csv_output_filepath, index=False)

    status = f"END: {fif_output_filepath.stem}"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(subjects)} Time elapsed: {time_elapsed} min"
    print(to_print)
