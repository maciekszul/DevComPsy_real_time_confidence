"""Simple least-squares removal of nuisance signals from MEG sensors."""

import numpy as np


def regress_signal(
    inst, ch_names=None, signal=None, picks="mag", zscore=False, copy=True
):
    """Regress one or more nuisance signals out of the sensors of a Raw/Epochs.

    An intercept is added to the model (a column of ones) and only the
    regressor part of the fit is subtracted, so each sensor keeps its mean.
    Everything is solved in one :func:`numpy.linalg.lstsq` call.

    Parameters
    ----------
    inst : instance of mne.io.Raw | mne.Epochs
        Data to clean.  Loaded into memory if needed.
    ch_names : str | sequence of str | None
        Channels of `inst` to use as regressors (e.g. EOG/ECG, head motion).
    signal : array | None
        Custom regressor(s) instead of `ch_names`: shape ``(n_times,)``,
        ``(n_regressors, n_times)`` or, for Epochs, ``(n_epochs, n_regressors,
        n_times)``.  Exactly one of `ch_names` / `signal` must be given.
    picks : str | sequence of str | array
        Channels to clean: a type understood by :func:`mne.pick_types`
        (``'mag'`` by default -- CTF axial gradiometers are typed ``mag`` --
        also ``'grad'``, ``'meg'``, ``'eeg'``), or explicit names/indices.
    zscore : bool
        Z-score each regressor independently before fitting.  The residuals are
        unchanged by this (an intercept is in the model, so the fitted subspace
        is the same); it only puts the betas on a comparable scale and helps
        conditioning when regressors differ wildly in magnitude.
    copy : bool
        Work on a copy (default) rather than modifying `inst` in place.

    Returns
    -------
    inst : instance of mne.io.Raw | mne.Epochs
        The same type as the input, with the regressors projected out.
    """
    from mne import pick_types

    if (ch_names is None) == (signal is None):
        raise ValueError("give exactly one of ch_names or signal")
    if copy:
        inst = inst.copy()
    inst.load_data()

    if ch_names is not None:
        ch_names = [ch_names] if isinstance(ch_names, str) else list(ch_names)
        conf = inst.get_data(picks=ch_names)
    else:
        conf = np.asarray(signal, dtype=np.float64)

    if isinstance(picks, str):
        sel = pick_types(inst.info, meg=picks if picks != "eeg" else False,
                         eeg=picks == "eeg", ref_meg=False)
    elif isinstance(picks[0], str):
        sel = np.array([inst.ch_names.index(p) for p in picks])
    else:
        sel = np.asarray(picks)

    data = inst._data  # Raw: (n_ch, n_times); Epochs: (n_epochs, n_ch, n_times)
    data3 = data[None] if data.ndim == 2 else data  # a view, writes through
    n_ep, _, n_t = data3.shape

    conf = conf.reshape((1,) * (3 - conf.ndim) + conf.shape)  # -> (?, n_reg, n_t)
    if conf.shape[-1] != n_t:
        raise ValueError(f"regressors must have {n_t} samples, got {conf.shape[-1]}")
    conf = np.broadcast_to(conf, (n_ep, conf.shape[1], n_t))

    # rows = all samples (epochs stacked), columns = [1, regressors]
    x = np.empty((n_ep * n_t, conf.shape[1] + 1))
    x[:, 0] = 1.0
    x[:, 1:] = conf.transpose(0, 2, 1).reshape(-1, conf.shape[1])
    if zscore:
        sd = x[:, 1:].std(axis=0)
        x[:, 1:] -= x[:, 1:].mean(axis=0)
        x[:, 1:] /= np.where(sd > 0, sd, 1.0)
    y = data3[:, sel].transpose(0, 2, 1).reshape(-1, len(sel))

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    y -= x[:, 1:] @ beta[1:]

    data3[:, sel] = y.reshape(n_ep, n_t, len(sel)).transpose(0, 2, 1)
    return inst
