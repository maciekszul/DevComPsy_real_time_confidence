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
from sklearn.svm import LinearSVC, NuSVR, SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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


def classify_binary_cv(data, target, pca=True, n_folds=10, n_jobs=10):
    if pca:
        pipeline = make_pipeline(
            RobustScaler(),
            PCA(n_components=0.95),
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
        cv=5,
        n_jobs=1
    )
    sliding_decoder = SlidingEstimator(
        grid, n_jobs=1, scoring="roc_auc"
    )
    scores = cross_val_multiscore(
        sliding_decoder, data, target, cv=n_folds, verbose=False, n_jobs=n_jobs)

    return scores


def classify_binary_nonlinear_cv(data, target, pca=True, n_folds=10, n_jobs=10):
    if pca:
        pipeline = make_pipeline(
            RobustScaler(),
            PCA(n_components=0.95),
            SVC(class_weight="balanced")
        )
    else:
        pipeline = make_pipeline(
            RobustScaler(),
            SVC(class_weight="balanced")
        )

    param_grid = {
        "svc__C": np.logspace(-3, 2, 6),
        "svc__gamma": np.logspace(-4, 1, 6),
        "svc__kernel": ["rbf"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=1
    )
    sliding_decoder = SlidingEstimator(
        grid, n_jobs=1, scoring="roc_auc"
    )
    scores = cross_val_multiscore(
        sliding_decoder, data, target, cv=n_folds, verbose=False, n_jobs=n_jobs)

    return scores


def classify_continuous_cv(data, target, n_folds=10, n_jobs=10):
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        NuSVR(
            max_iter=10000
        )
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
        cv=5,
        n_jobs=1
    )
    sliding_decoder = SlidingEstimator(
        grid, n_jobs=1, scoring="r2"
    )

    scores = cross_val_multiscore(
        sliding_decoder, data, target, cv=n_folds, verbose=False, n_jobs=n_jobs)

    return scores




def prepare_data(subject, tmin=None, tmax=None, fmin=None, fmax=None, hilbert_envelope=False, decimate=2, baseline=False):
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
        subject, tmin=None, tmax=None, fmin=None, fmax=None, hilbert_envelope=False, baseline=(-0.25, 0.0)
    )
    # classification ranked confidence
    # target = behaviour.confidence_rank.to_numpy()
    # scores = classify_continuous_cv(data, target, n_folds=10, n_jobs=1)
    # class_name = "rank_confidence_sliding_nusvr_cv"
    n_jobs = 3
    # binary classifications
    target = behaviour.confidence_rank.to_numpy() >= 0.5

    # broadband
    bandpass = "broadband"
    clas_func = classify_binary_cv
    scores = clas_func(
        data, target, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)
    
    del data



    # alpha
    times, info, data, behaviour, subj_path = prepare_data(
        subject, tmin=None, tmax=None, fmin=8, fmax=13, hilbert_envelope=False, baseline=(-0.25, 0.0)
    )

    bandpass = "alpha"
    clas_func = classify_binary_cv
    scores = clas_func(
        data, target, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del data



    bandpass = "alpha_ssd"
    
    times, info, data, behaviour, subj_path = prepare_data(
        subject, tmin=None, tmax=None, fmin=None, fmax=None, hilbert_envelope=False, baseline=(-0.25, 0.0)
    )
    ssd_file = utils.get_files(proc_path, "*.pkl", strings=[subject])[0]
    ssd = utils.load_pickle(ssd_file)
    data = ssd.transform(data)

    clas_func = classify_binary_cv
    scores = clas_func(
        data, target, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    phase, amp, freq = apply_func_fast_multi(data, extract_insta, sfreq=info["sfreq"])
    del data



    bandpass = "alpha_ssd_phase"

    clas_func = classify_binary_cv
    scores = clas_func(
        phase, target, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del phase



    bandpass = "alpha_ssd_amplitude"

    clas_func = classify_binary_cv
    scores = clas_func(
        amp, target, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del amp



    bandpass = "alpha_ssd_frequency"

    clas_func = classify_binary_cv
    scores = clas_func(
        freq, target, pca=False, n_folds=10, n_jobs=n_jobs
    )
    class_name = f"binary_confidence-{clas_func.__name__}-{bandpass}"
    scores_path = subj_path.joinpath(f"scores-{class_name}-{subject}.npy")
    np.save(scores_path, scores, allow_pickle=False)
    print("SAVED", scores_path)

    del freq

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(subjects)} Time elapsed: {time_elapsed} min"
    print(to_print)