import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import utils
import warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tools.shape import apply_func_fast_multi, extract_insta
from pathlib import Path
from mne import (
    set_log_level,
    read_epochs,
    concatenate_epochs
)

from mne.decoding import (
    SlidingEstimator, GeneralizingEstimator,
    cross_val_multiscore
)

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC, NuSVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV

set_log_level("CRITICAL")

warnings.filterwarnings('ignore')


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


def residualise(data, labels):
    resid_data = []
    for t in range(data.shape[-1]):
        resid = LinearRegression()
        resid.fit(labels, data[:,:, t])
        resid_data.append(data[:,:, t] - resid.predict(labels))
    return np.stack(resid_data, axis=-1)

# def residualise(data, x):
#     """
#     Residualize data (trials × sensors × time) with respect to x (trials,).

#     NaNs in data are replaced with the minimum finite value.
#     """
#     data = data.copy()

#     # Replace NaNs with global minimum finite value
#     data[np.isnan(data)] = np.nanmin(data)

#     X = np.column_stack((np.ones_like(x), x))
#     Y = data.reshape(len(x), -1)

#     beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
#     return (Y - X @ beta).reshape(data.shape)


def search_settings_binary(data, labels, pca=True):
    if pca:
        pipeline = make_pipeline(
            RobustScaler(),
            PCA(n_components=15),
            LinearSVC(
                max_iter=10000,
                class_weight="balanced"
            )
        )
    else:
        pipeline = make_pipeline(
            RobustScaler(),
            LinearSVC(
                max_iter=10000,
                class_weight="balanced"
            )
        )

    param_grid = {
        "linearsvc__penalty": ["l1", "l2"],
        "linearsvc__C": np.logspace(-3, 2, 6)
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        n_jobs=10
    )

    param_range = []
    for t in range(data.shape[-1]):
        features = data[:,:, t]
        grid.fit(features, labels)
        param_range.append(grid.best_estimator_.get_params())
    
    settings = {}
    for k in param_grid.keys():
        params = [i[k] for i in param_range]
        params = np.unique_counts(params)
        settings[k] = params.values[params.counts.argmax()]
    
    return settings


def classify_binary(data, target, settings, pca=True, n_folds=10, n_jobs=10):
    if pca:
        pipeline = make_pipeline(
            RobustScaler(),
            PCA(n_components=15),
            LinearSVC(class_weight="balanced")
        )
    else:
        pipeline = make_pipeline(
            RobustScaler(),
            LinearSVC(class_weight="balanced")
        )

    pipeline.set_params(**settings)

    sliding_decoder = SlidingEstimator(
        pipeline, n_jobs=1, scoring="roc_auc"
    )
    scores = cross_val_multiscore(
        sliding_decoder, data, target, cv=n_folds, verbose=False, n_jobs=n_jobs)

    return scores


def search_settings_continuous(data, labels, pca=True):
    if pca:
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=15),
            NuSVR(max_iter=10000)
        )
    else:
        pipeline = make_pipeline(
            StandardScaler(),
            NuSVR(max_iter=10000)
        )

    param_grid = {
        "nusvr__C": [0.1, 1, 10],
        "nusvr__nu": [0.25, 0.5, 0.75],
        "nusvr__kernel": ["linear", "rbf"],
        "nusvr__gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        n_jobs=10
    )

    param_range = []
    for t in range(data.shape[-1]):
        features = data[:,:, t]
        grid.fit(features, labels)
        param_range.append(grid.best_estimator_.get_params())
    
    settings = {}
    for k in param_grid.keys():
        params = [i[k] for i in param_range]
        params = np.unique_counts(params)
        settings[k] = params.values[params.counts.argmax()]
    
    return settings


def classify_continuous(data, target, settings, pca=True, n_folds=10, n_jobs=10):
    if pca:
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=15),
            NuSVR(max_iter=10000)
        )
    else:
        pipeline = make_pipeline(
            StandardScaler(),
            NuSVR(max_iter=10000)
        )
    
    pipeline.set_params(**settings)
    
    sliding_decoder = SlidingEstimator(
        pipeline, n_jobs=1, scoring="r2"
    )

    scores = cross_val_multiscore(
        sliding_decoder, data, target, cv=n_folds, verbose=False, n_jobs=n_jobs
    )

    return scores


def prepare_data(subject, tmin=None, tmax=None, fmin=None, fmax=None, hilbert_envelope=False, decimate=1, baseline=False):
    beh_files = utils.get_files(beh_path, "*.csv", strings=["main_", subject])
    beh_files = [i for i in beh_files if "ses_0" not in i._str]
    fif_files = utils.get_files(proc_path, "*.fif", strings=["dots-onset", "realtime", "epo.fif", subject])
    fif_files = [i for i in fif_files if "ses_0" not in i._str]
    meg_calib_file = utils.get_files(proc_path, "*.json", strings=["meg-calibration", subject])[0]
    meg_calib = utils.load_json(meg_calib_file)

    behaviour = pd.concat([pd.read_csv(i) for i in beh_files]).reset_index(drop=True)
    behaviour = behaviour[[i for i in behaviour.columns.tolist() if "exp_timing" not in i]]
    epochs = concatenate_epochs([read_epochs(i, verbose=False) for i in fif_files], add_offset=True, on_mismatch="ignore", verbose=False)
    overlap = np.mean(epochs.events[:,2] == behaviour.trial_trigger.to_numpy()) * 100
    print(f"epoch order overlaps in {overlap}%")

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
    if not fmin == fmax:
        epochs = epochs.filter(fmin, fmax)
    if hilbert_envelope:
        epochs = epochs.apply_hilbert(envelope=True)
    if baseline:
        epochs = epochs.apply_baseline(baseline)
    epochs = epochs.decimate(decimate)
    epochs = epochs.apply_baseline((-.25, 0))
    epochs = epochs.crop(tmin, tmax)
    epochs = epochs.pick(picks="mag")
    times = epochs.times
    info = epochs.info
    epochs = epochs.get_data()   
    return times, info, epochs, behaviour, meg_calib_file.parent


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
    times, info, data, behaviour, subj_path = prepare_data(
        subject, tmin=None, tmax=None, fmin=None, fmax=None, hilbert_envelope=False, baseline=(-0.25, 0.0), decimate=2
    )
    # classification ranked confidence

    n_jobs = 3
    search_n = 3
    # binary classifications
    
    ACC = behaviour.response_correct.to_numpy().reshape(-1,1).astype(bool)
    ACC = np.nan_to_num(ACC, nan=False)
    CONF = behaviour.confidence_rank.to_numpy().reshape(-1,1).astype(float)
    CONF = np.nan_to_num(ACC, nan=0.0)
    CONF_BIN = CONF > 0.5

    

    # data = residualise(data, ACC)

    # broadband
    bandpass = "broadband"

    settings = search_settings_binary(data[:,:, ::search_n], CONF_BIN.flatten())
    clas_func = classify_binary
    scores = clas_func(
        data, CONF_BIN.flatten(), settings, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"residualised_binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)
    
    del data



    # alpha
    times, info, data, behaviour, subj_path = prepare_data(
        subject, tmin=None, tmax=None, fmin=8, fmax=13, hilbert_envelope=False, baseline=(-0.25, 0.0), decimate=2
    )

    # data = residualise(data, ACC)

    bandpass = "alpha"

    settings = search_settings_binary(data[:,:, ::search_n], CONF_BIN.flatten())
    clas_func = classify_binary
    scores = clas_func(
        data, CONF_BIN.flatten(), settings, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"residualised_binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del data



    bandpass = "alpha_ssd"
    
    times, info, data, behaviour, subj_path = prepare_data(
        subject, tmin=None, tmax=None, fmin=None, fmax=None, hilbert_envelope=False, baseline=(-0.25, 0.0), decimate=2
    )

    ssd_file = utils.get_files(proc_path, "*.pkl", strings=[subject])[0]
    ssd = utils.load_pickle(ssd_file)
    data = ssd.transform(data)

    # data = residualise(data, ACC)

    settings = search_settings_binary(data[:,:, ::search_n], CONF_BIN.flatten(), pca=False)
    clas_func = classify_binary
    scores = clas_func(
        data, CONF_BIN.flatten(), settings, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"residualised_binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    phase, amp, freq = apply_func_fast_multi(data, extract_insta, sfreq=info["sfreq"])
    del data



    bandpass = "alpha_ssd_phase"

    settings = search_settings_binary(phase[:,:, ::search_n], CONF_BIN.flatten(), pca=False)
    clas_func = classify_binary
    scores = clas_func(
        phase, CONF_BIN.flatten(), settings, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"residualised_binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del phase



    bandpass = "alpha_ssd_amplitude"

    settings = search_settings_binary(amp[:,:, ::search_n], CONF_BIN.flatten(), pca=False)
    clas_func = classify_binary
    scores = clas_func(
        amp, CONF_BIN.flatten(), settings, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"residualised_binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del amp



    bandpass = "alpha_ssd_frequency"

    settings = search_settings_binary(freq[:,:, ::search_n], CONF_BIN.flatten(), pca=False)
    clas_func = classify_binary
    scores = clas_func(
        freq, CONF_BIN.flatten(), settings, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"residualised_binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del freq

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(subjects)} Time elapsed: {time_elapsed} min"
    print(to_print)