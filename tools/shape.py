import emd
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

"""
Assumption of band-passed signal.
"""


def extract_shape_trial(signal, time, sfreq, imf_ix=0, mm_amp_thr=0.05, trial_label=0, spatial_label="ssd", n_points=48):
    imf = emd.sift.sift(signal, max_imfs=1)
    IP, IF, IA = emd.spectra.frequency_transform(imf, sfreq, 'nht')
    mask = minmax_scale(IA[:,imf_ix]) > mm_amp_thr
    cycles = emd.cycles.get_cycle_vector(IP[:,imf_ix], return_good=True, mask=mask)
    labels = ["mode", "amp", "freq"]
    output_dfs = []
    peaks_loc = []
    troughs_loc = []
    for lab_ix, metric in enumerate([imf, IA, IF]):
        label = labels[lab_ix]
        pa_metric, phase_template = emd.cycles.phase_align(
            IP[:, imf_ix],
            metric[:, imf_ix],
            cycles=cycles[:, imf_ix],
            npoints=n_points
        )
        pa_metric = pd.DataFrame(pa_metric.T)
        pa_metric.columns = [f"{label}_{str(i).zfill(3)}" for i in range(pa_metric.shape[1])]
        output_dfs.append(pa_metric)
        
        if label == "mode":
            peaks_loc = np.argmax(pa_metric, axis=1)
            troughs_loc = np.argmin(pa_metric, axis=1)
    
    unique_cycles = np.unique(cycles)
    unique_cycles = unique_cycles[unique_cycles != -1]
    control_points = emd.cycles.get_control_points(imf[:, imf_ix], cycles.flatten())
    
    time_features = {
        "duration": [],
        "onset": [],
        "time_peak_abs": [],
        "time_peak_rel": [],
        "time_trough_abs": [],
        "time_trough_rel": [],
        "amp_peak": [],
        "amp_trough": []
    }
    for ix_uc, uc in enumerate(unique_cycles):
        cycle_map = cycles.flatten() == uc
        control_point = control_points[ix_uc]
        timed_cp = time[cycle_map][control_point]
        zeroed_cp = timed_cp - timed_cp[0]
        amps = imf[:, imf_ix][control_point]
        time_features["duration"].append(zeroed_cp[-1])
        time_features["onset"].append(timed_cp[0])
        time_features["time_peak_abs"].append(timed_cp[1])
        time_features["time_peak_rel"].append(zeroed_cp[1])
        time_features["time_trough_abs"].append(timed_cp[4])
        time_features["time_trough_rel"].append(zeroed_cp[4])
        time_features["amp_peak"].append(amps[1])
        time_features["amp_trough"].append(amps[4])


    time_features = pd.DataFrame.from_dict(time_features)
    time_features["pa_peak_ix"] = peaks_loc
    time_features["pa_trough_ix"] = troughs_loc
    output_dfs.append(time_features)
    output_dfs = pd.concat(output_dfs, axis=1)
    output_dfs = output_dfs.assign(trial=trial_label)
    output_dfs = output_dfs.assign(spatial_label=spatial_label)


    return output_dfs


def extract_insta(signal, sfreq, imf_ix=0, get=None):
    """
    data in trial x time shape
    """
    imf = emd.sift.sift(signal, max_imfs=1)
    imf = imf[:, imf_ix].flatten()
    phase, frequency, amplitude = emd.spectra.frequency_transform(imf, sfreq, 'nht')

    if get == "phase":
        return phase.flatten()
    elif get == "frequency":
        return frequency.flatten()
    elif get == "amplitude":
        return amplitude.flatten()
    else:
        return phase.flatten(), amplitude.flatten(), frequency.flatten()


def apply_func_fast(arr, func, *args, **kwargs):
    n1, n2, n3 = arr.shape
    reshaped = arr.reshape(-1, n3)

    out = np.empty_like(reshaped)
    for i, row in enumerate(reshaped):
        out[i] = func(row, *args, **kwargs)

    return out.reshape(n1, n2, n3)


def apply_func_fast_multi(arr, func, *args, **kwargs):
    n1, n2, n3 = arr.shape
    reshaped = arr.reshape(-1, n3)

    # Run once to infer number of outputs
    first = func(reshaped[0], *args, **kwargs)
    n_out = len(first)

    outs = [np.empty_like(reshaped) for _ in range(n_out)]

    # fill first row
    for j in range(n_out):
        outs[j][0] = first[j]

    # loop remaining
    for i in range(1, reshaped.shape[0]):
        results = func(reshaped[i], *args, **kwargs)
        for j in range(n_out):
            outs[j][i] = results[j]

    # reshape back
    return tuple(o.reshape(n1, n2, n3) for o in outs)


def apply_func_along_axis(arr, axis, func, *args, **kwargs):
    arr = np.asarray(arr)
    
    # Move target axis to last for easier iteration
    arr_moved = np.moveaxis(arr, axis, -1)
    shape = arr_moved.shape
    n = shape[-1]

    flat = arr_moved.reshape(-1, n)

    # infer number of outputs
    first = func(flat[0], *args, **kwargs)
    n_out = len(first)

    outs = [
        np.empty((flat.shape[0], n), dtype=np.asarray(first[i]).dtype)
        for i in range(n_out)
    ]

    # first element
    for i, r in enumerate(first):
        outs[i][0] = r

    # main loop
    for i in range(1, flat.shape[0]):
        results = func(flat[i], *args, **kwargs)
        for j, r in enumerate(results):
            outs[j][i] = r

    # reshape back
    out_shapes = [
        np.moveaxis(o.reshape(shape), -1, axis)
        for o in outs
    ]

    return tuple(out_shapes)