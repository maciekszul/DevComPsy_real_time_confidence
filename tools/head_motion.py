"""
CTF head-localisation coils (HLC channels) -> 6 DoF rigid-body head motion.

The CTF continuous head localisation stores the x/y/z position of each coil in
dewar (device) coordinates as separate channels:

    HLC0011 HLC0012 HLC0013 -> nasion x, y, z
    HLC0021 HLC0022 HLC0023 -> LPA    x, y, z
    HLC0031 HLC0032 HLC0033 -> RPA    x, y, z

(mapping per mne.chpi.extract_chpi_locs_ctf and FieldTrip ft_headmovement.m;
units are metres once MNE/FieldTrip calibration has been applied).

It also hosts the tSSS/MaxFilter side of head motion, which deals with the
same problem by a different route: instead of removing motion-correlated
variance after the fact, SSS re-renders every sample as if the head had been
at one fixed position.

Main entry points
-----------------
fiducials_to_6dof : (nasion, lpa, rpa) time series -> (n_times, 6) motion
raw_to_6dof       : appends the motion to an MNE Raw as misc channels
regress_head_motion : regresses those 6 confounds out of the MEG sensors
apply_tsss        : tSSS + movement compensation of a continuous CTF Raw
ctf_head_positions : HLC channels -> the head_pos array maxwell_filter wants
"""

import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is in the env, this is a safety net
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f

        return wrap(args[0]) if args and callable(args[0]) else wrap

    prange = range


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _frames_from_fiducials(nas, lpa, rpa, origin, rot):
    """Per-sample orthonormal head frame (CTF convention) from 3 fiducials.

    x -> nasion, y -> left (LPA), z = x cross y.  `rot` columns are the head
    axes expressed in device coordinates, i.e. p_dev = rot @ p_head + origin.
    """
    n = nas.shape[0]
    for i in prange(n):
        ox = 0.5 * (lpa[i, 0] + rpa[i, 0])
        oy = 0.5 * (lpa[i, 1] + rpa[i, 1])
        oz = 0.5 * (lpa[i, 2] + rpa[i, 2])
        origin[i, 0] = ox
        origin[i, 1] = oy
        origin[i, 2] = oz

        # x axis: origin -> nasion
        xx = nas[i, 0] - ox
        xy = nas[i, 1] - oy
        xz = nas[i, 2] - oz
        nrm = np.sqrt(xx * xx + xy * xy + xz * xz)
        xx /= nrm
        xy /= nrm
        xz /= nrm

        # provisional y axis: RPA -> LPA, orthogonalised against x
        yx = lpa[i, 0] - rpa[i, 0]
        yy = lpa[i, 1] - rpa[i, 1]
        yz = lpa[i, 2] - rpa[i, 2]
        d = yx * xx + yy * xy + yz * xz
        yx -= d * xx
        yy -= d * xy
        yz -= d * xz
        nrm = np.sqrt(yx * yx + yy * yy + yz * yz)
        yx /= nrm
        yy /= nrm
        yz /= nrm

        # z = x cross y
        zx = xy * yz - xz * yy
        zy = xz * yx - xx * yz
        zz = xx * yy - xy * yx

        rot[i, 0, 0] = xx
        rot[i, 1, 0] = xy
        rot[i, 2, 0] = xz
        rot[i, 0, 1] = yx
        rot[i, 1, 1] = yy
        rot[i, 2, 1] = yz
        rot[i, 0, 2] = zx
        rot[i, 1, 2] = zy
        rot[i, 2, 2] = zz


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _relative_6dof(origin, rot, o_ref, r_ref, in_ref_frame, rotvec, out):
    """Motion of each sample's head frame w.r.t. the reference frame.

    R_rel = R_ref.T @ R_t,  t_rel = R_ref.T @ (o_t - o_ref)   (reference frame)
    or     R_rel = R_t @ R_ref.T,  t_rel = o_t - o_ref        (device frame)
    """
    n = origin.shape[0]
    for i in prange(n):
        if in_ref_frame:
            r00 = r_ref[0, 0] * rot[i, 0, 0] + r_ref[1, 0] * rot[i, 1, 0] + r_ref[2, 0] * rot[i, 2, 0]
            r01 = r_ref[0, 0] * rot[i, 0, 1] + r_ref[1, 0] * rot[i, 1, 1] + r_ref[2, 0] * rot[i, 2, 1]
            r02 = r_ref[0, 0] * rot[i, 0, 2] + r_ref[1, 0] * rot[i, 1, 2] + r_ref[2, 0] * rot[i, 2, 2]
            r10 = r_ref[0, 1] * rot[i, 0, 0] + r_ref[1, 1] * rot[i, 1, 0] + r_ref[2, 1] * rot[i, 2, 0]
            r11 = r_ref[0, 1] * rot[i, 0, 1] + r_ref[1, 1] * rot[i, 1, 1] + r_ref[2, 1] * rot[i, 2, 1]
            r12 = r_ref[0, 1] * rot[i, 0, 2] + r_ref[1, 1] * rot[i, 1, 2] + r_ref[2, 1] * rot[i, 2, 2]
            r20 = r_ref[0, 2] * rot[i, 0, 0] + r_ref[1, 2] * rot[i, 1, 0] + r_ref[2, 2] * rot[i, 2, 0]
            r21 = r_ref[0, 2] * rot[i, 0, 1] + r_ref[1, 2] * rot[i, 1, 1] + r_ref[2, 2] * rot[i, 2, 1]
            r22 = r_ref[0, 2] * rot[i, 0, 2] + r_ref[1, 2] * rot[i, 1, 2] + r_ref[2, 2] * rot[i, 2, 2]
            dx = origin[i, 0] - o_ref[0]
            dy = origin[i, 1] - o_ref[1]
            dz = origin[i, 2] - o_ref[2]
            out[i, 0] = r_ref[0, 0] * dx + r_ref[1, 0] * dy + r_ref[2, 0] * dz
            out[i, 1] = r_ref[0, 1] * dx + r_ref[1, 1] * dy + r_ref[2, 1] * dz
            out[i, 2] = r_ref[0, 2] * dx + r_ref[1, 2] * dy + r_ref[2, 2] * dz
        else:
            r00 = rot[i, 0, 0] * r_ref[0, 0] + rot[i, 0, 1] * r_ref[0, 1] + rot[i, 0, 2] * r_ref[0, 2]
            r01 = rot[i, 0, 0] * r_ref[1, 0] + rot[i, 0, 1] * r_ref[1, 1] + rot[i, 0, 2] * r_ref[1, 2]
            r02 = rot[i, 0, 0] * r_ref[2, 0] + rot[i, 0, 1] * r_ref[2, 1] + rot[i, 0, 2] * r_ref[2, 2]
            r10 = rot[i, 1, 0] * r_ref[0, 0] + rot[i, 1, 1] * r_ref[0, 1] + rot[i, 1, 2] * r_ref[0, 2]
            r11 = rot[i, 1, 0] * r_ref[1, 0] + rot[i, 1, 1] * r_ref[1, 1] + rot[i, 1, 2] * r_ref[1, 2]
            r12 = rot[i, 1, 0] * r_ref[2, 0] + rot[i, 1, 1] * r_ref[2, 1] + rot[i, 1, 2] * r_ref[2, 2]
            r20 = rot[i, 2, 0] * r_ref[0, 0] + rot[i, 2, 1] * r_ref[0, 1] + rot[i, 2, 2] * r_ref[0, 2]
            r21 = rot[i, 2, 0] * r_ref[1, 0] + rot[i, 2, 1] * r_ref[1, 1] + rot[i, 2, 2] * r_ref[1, 2]
            r22 = rot[i, 2, 0] * r_ref[2, 0] + rot[i, 2, 1] * r_ref[2, 1] + rot[i, 2, 2] * r_ref[2, 2]
            out[i, 0] = origin[i, 0] - o_ref[0]
            out[i, 1] = origin[i, 1] - o_ref[1]
            out[i, 2] = origin[i, 2] - o_ref[2]

        if rotvec:
            # axis-angle (rotation vector), via the matrix logarithm
            tr = r00 + r11 + r22
            c = 0.5 * (tr - 1.0)
            if c > 1.0:
                c = 1.0
            elif c < -1.0:
                c = -1.0
            ang = np.arccos(c)
            ax = r21 - r12
            ay = r02 - r20
            az = r10 - r01
            s = np.sqrt(ax * ax + ay * ay + az * az)
            if s < 1e-12:
                # ~0 or ~pi; small-angle branch is the relevant one for head motion
                out[i, 3] = 0.5 * ax
                out[i, 4] = 0.5 * ay
                out[i, 5] = 0.5 * az
            else:
                k = ang / s
                out[i, 3] = k * ax
                out[i, 4] = k * ay
                out[i, 5] = k * az
        else:
            # intrinsic X-Y-Z Euler angles: R = Rx(rx) @ Ry(ry) @ Rz(rz)
            sy = r02
            if sy > 1.0:
                sy = 1.0
            elif sy < -1.0:
                sy = -1.0
            if np.abs(sy) < 0.999999:
                out[i, 3] = np.arctan2(-r12, r22)
                out[i, 4] = np.arcsin(sy)
                out[i, 5] = np.arctan2(-r01, r00)
            else:  # gimbal lock
                out[i, 3] = np.arctan2(r21, r11)
                out[i, 4] = np.arcsin(sy)
                out[i, 5] = 0.0


def fiducials_to_6dof(
    nasion,
    lpa,
    rpa,
    reference=0,
    rotation="euler",
    degrees=True,
    frame="reference",
    method="frame",
    scale=1.0,
):
    """Convert fiducial coil trajectories into 6 DoF rigid-body head motion.

    Parameters
    ----------
    nasion, lpa, rpa : array, shape (n_times, 3)
        Coil positions over time (device/dewar coordinates, metres for CTF).
        Anything array-like works; float64 C-contiguous is fastest.
    reference : int | 'mean' | array, shape (3, 3)
        Head position that motion is expressed relative to: a sample index
        (default 0, i.e. the first sample), ``'mean'`` for the across-time mean
        fiducial positions, or an explicit ``[nasion, lpa, rpa]`` array.
    rotation : 'euler' | 'rotvec'
        ``'euler'`` -> intrinsic X-Y-Z angles (scipy ``as_euler('XYZ')``),
        ``'rotvec'`` -> axis-angle vector (scipy ``as_rotvec()``).
    degrees : bool
        Return angles in degrees (default) rather than radians.
    frame : 'reference' | 'device'
        ``'reference'`` expresses translation/rotation in the reference head
        frame (the usual motion-parameter convention, as in fMRI realignment);
        ``'device'`` keeps them in dewar coordinates.
    method : 'frame' | 'kabsch'
        ``'frame'`` builds the head axes directly from the three coils (CTF
        convention: x->nasion, y->left, z=x*y) -- exact and fastest.
        ``'kabsch'`` least-squares fits each sample's triangle to the reference
        triangle (SVD), which spreads the error if a coil slips.
    scale : float
        Multiplier applied to the translations, e.g. ``1000`` to go m -> mm.

    Returns
    -------
    motion : ndarray, shape (n_times, 6)
        ``[tx, ty, tz, rx, ry, rz]`` -- translation in input units * `scale`,
        rotation in degrees (or radians).  Row `reference` is all zeros when an
        index is used.

    Notes
    -----
    ~1e6 samples take well under a second; the heavy lifting is a single
    parallel numba pass with no temporaries.
    """
    nas = np.ascontiguousarray(nasion, dtype=np.float64)
    lp = np.ascontiguousarray(lpa, dtype=np.float64)
    rp = np.ascontiguousarray(rpa, dtype=np.float64)
    if nas.shape != lp.shape or nas.shape != rp.shape or nas.ndim != 2 or nas.shape[1] != 3:
        raise ValueError("nasion, lpa and rpa must all have shape (n_times, 3)")
    if rotation not in ("euler", "rotvec"):
        raise ValueError("rotation must be 'euler' or 'rotvec'")
    if frame not in ("reference", "device"):
        raise ValueError("frame must be 'reference' or 'device'")

    ref = _reference_fiducials(nas, lp, rp, reference)

    if method == "kabsch":
        origin, rot = _kabsch_frames(np.stack([nas, lp, rp], axis=1), ref)
        o_ref = ref.mean(axis=0)
        r_ref = np.eye(3)
    elif method == "frame":
        n = nas.shape[0]
        origin = np.empty((n, 3))
        rot = np.empty((n, 3, 3))
        _frames_from_fiducials(nas, lp, rp, origin, rot)
        o_ref = np.empty(3)
        r_ref = np.empty((1, 3, 3))
        _frames_from_fiducials(
            ref[0][None], ref[1][None], ref[2][None], o_ref[None], r_ref
        )
        r_ref = r_ref[0]
    else:
        raise ValueError("method must be 'frame' or 'kabsch'")

    out = np.empty((nas.shape[0], 6))
    _relative_6dof(
        origin,
        rot,
        o_ref,
        r_ref,
        frame == "reference",
        rotation == "rotvec",
        out,
    )

    if degrees:
        out[:, 3:] *= 180.0 / np.pi
    if scale != 1.0:
        out[:, :3] *= scale
    return out


HEAD_MOTION_CH_NAMES = (
    "head_trans_x",
    "head_trans_y",
    "head_trans_z",
    "head_rot_x",
    "head_rot_y",
    "head_rot_z",
)


def raw_to_6dof(raw, picks=None, ch_names=HEAD_MOTION_CH_NAMES, **kwargs):
    """Append the 6 DoF head motion of an MNE Raw's HLC channels to that Raw.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        CTF recording containing the 9 HLC continuous head-localisation
        channels.  Loaded into memory if it is not already, and modified in
        place.
    picks : None | array, shape (9,)
        Channel indices ordered nasion x/y/z, LPA x/y/z, RPA x/y/z.  Found
        automatically from the ``HLC00[123][123]`` names when None.
    ch_names : sequence of str, len 6
        Labels of the appended misc channels, in the order
        ``tx, ty, tz, rx, ry, rz``.  Channels of the same name already present
        are replaced, so the function can be re-run with different settings.
    **kwargs
        Passed to :func:`fiducials_to_6dof` (``reference``, ``rotation``,
        ``degrees``, ``frame``, ``method``, ``scale``).

    Returns
    -------
    raw : instance of mne.io.Raw
        The same object, with the 6 DoF time courses appended as misc channels.

    Notes
    -----
    The channels hold whatever units :func:`fiducials_to_6dof` produced (metres
    and degrees by default), so a FIFF unit is tagged only where it is
    unambiguous: metres for the translations when ``scale == 1``, radians for
    the rotations when ``degrees=False``.  The settings used are recorded in
    ``raw.info['description']``.
    """
    from mne import create_info, pick_channels_regexp
    from mne.io import RawArray
    from mne.io.constants import FIFF

    ch_names = list(ch_names)
    if len(ch_names) != 6:
        raise ValueError(f"ch_names must have 6 entries, got {len(ch_names)}")
    if picks is None:
        picks = pick_channels_regexp(raw.info["ch_names"], "HLC00[123][123].*")
        if len(picks) != 9:
            raise RuntimeError(f"Expected 9 HLC channels, found {len(picks)}")
        # alphabetical -> HLC0011..13 (nasion), 0021..23 (LPA), 0031..33 (RPA)
        picks = np.array(sorted(picks, key=lambda k: raw.info["ch_names"][k]))

    raw.load_data()
    data = raw[picks, :][0].T.reshape(-1, 3, 3)
    motion = fiducials_to_6dof(data[:, 0], data[:, 1], data[:, 2], **kwargs)

    dupes = [ch for ch in ch_names if ch in raw.ch_names]
    if dupes:
        raw.drop_channels(dupes)

    degrees = kwargs.get("degrees", True)
    scale = kwargs.get("scale", 1.0)
    info = create_info(ch_names, raw.info["sfreq"], ["misc"] * 6)
    with info._unlock():
        info["meas_date"] = raw.info["meas_date"]
        for i, ch in enumerate(info["chs"]):
            if i < 3 and scale == 1.0:
                ch["unit"] = FIFF.FIFF_UNIT_M
            elif i >= 3 and not degrees:
                ch["unit"] = FIFF.FIFF_UNIT_RAD
    extra = RawArray(
        np.ascontiguousarray(motion.T), info, first_samp=raw.first_samp, verbose="error"
    )
    raw.add_channels([extra], force_update_info=True)

    note = (
        f"head motion 6 DoF ({', '.join(ch_names)}); "
        f"translations in {'m' if scale == 1.0 else f'm * {scale:g}'}, "
        f"rotations in {'deg' if degrees else 'rad'}"
    )
    desc = raw.info["description"]
    with raw.info._unlock():
        raw.info["description"] = note if not desc else f"{desc} | {note}"
    return raw


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reference_fiducials(nas, lpa, rpa, reference):
    """Resolve `reference` into a (3, 3) array of [nasion, lpa, rpa]."""
    if isinstance(reference, str):
        if reference != "mean":
            raise ValueError("string reference must be 'mean'")
        return np.stack([nas.mean(0), lpa.mean(0), rpa.mean(0)])
    if np.isscalar(reference):
        i = int(reference)
        return np.stack([nas[i], lpa[i], rpa[i]])
    ref = np.asarray(reference, dtype=np.float64)
    if ref.shape != (3, 3):
        raise ValueError("explicit reference must have shape (3, 3)")
    return ref


def _kabsch_frames(pts, ref):
    """Least-squares rigid fit of each sample's 3 coils onto the reference.

    Returns the per-sample centroid and the rotation taking reference-centred
    coordinates to sample coordinates.
    """
    ref_c = ref - ref.mean(axis=0)
    origin = pts.mean(axis=1)
    p = pts - origin[:, None, :]
    h = np.einsum("nij,ik->njk", p, ref_c)  # (n, 3, 3) covariance
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    u[:, :, -1] *= d[:, None]
    rot = np.ascontiguousarray(u @ vt)
    return origin, rot


def head_motion_summary(motion):
    """Peak-to-peak, RMS and max absolute displacement per DoF."""
    return {
        "ptp": np.ptp(motion, axis=0),
        "rms": np.sqrt(np.mean(motion**2, axis=0)),
        "max_abs": np.max(np.abs(motion), axis=0),
    }


def regress_head_motion(
    epochs, ch_names=HEAD_MOTION_CH_NAMES, mode="trial", picks="meg", copy=True
):
    """Regress the 6 DoF head motion out of the MEG sensors of an Epochs object.

    The 6 confounds are z-scored independently, an intercept column of ones is
    added, and the least-squares fit is solved once for all sensors/time points
    with :func:`numpy.linalg.lstsq`.  Only the confound part of the fit is
    subtracted, so each sensor keeps its mean (the intercept).

    Parameters
    ----------
    epochs : instance of mne.Epochs
        Must contain the 6 head-motion channels appended by
        :func:`raw_to_6dof` (before epoching).
    ch_names : sequence of str, len 6
        Names of those channels.
    mode : 'trial' | 'sample'
        ``'trial'`` (default) regresses across trials, using each trial's mean
        head position -- one beta per sensor and time point, i.e. the usual
        head-motion confound regression (cf. FieldTrip ``ft_regressconfound``).
        ``'sample'`` regresses across all trials *and* time points pooled, using
        the instantaneous head position -- one beta per sensor.
    picks : str | array
        Channels to clean, anything :func:`mne.pick_types`-like accepts via
        ``epochs.get_data(picks=...)``; defaults to MEG sensors only
        (reference channels excluded).
    copy : bool
        Work on a copy (default) instead of modifying `epochs` in place.

    Returns
    -------
    epochs : instance of mne.Epochs
        With the head-motion variance removed from the picked sensors.
    """
    from mne import pick_types

    if copy:
        epochs = epochs.copy()
    epochs.load_data()

    ch_names = list(ch_names)
    missing = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"Head-motion channels missing from epochs: {missing}")

    if isinstance(picks, str) and picks == "meg":
        sel = pick_types(epochs.info, meg=True, ref_meg=False)
    else:
        sel = np.asarray(
            [epochs.ch_names.index(p) for p in picks]
            if isinstance(picks[0], str)
            else picks
        )
    conf = epochs.get_data(picks=ch_names)  # (n_epochs, 6, n_times)
    data = epochs._data[:, sel]  # (n_epochs, n_sel, n_times)
    n_ep, n_sel, n_t = data.shape

    if mode == "trial":
        x = conf.mean(axis=2)  # (n_epochs, 6)
        y = data.reshape(n_ep, n_sel * n_t)
    elif mode == "sample":
        x = conf.transpose(0, 2, 1).reshape(n_ep * n_t, 6)
        y = data.transpose(0, 2, 1).reshape(n_ep * n_t, n_sel)
    else:
        raise ValueError("mode must be 'trial' or 'sample'")

    # z-score each confound independently, then add the intercept column
    sd = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    x = np.column_stack([np.ones(len(x)), x])

    beta = np.linalg.lstsq(x, y, rcond=None)[0]  # (7, n_targets)
    y = y - x[:, 1:] @ beta[1:]  # keep the intercept in the data

    if mode == "trial":
        data[:] = y.reshape(n_ep, n_sel, n_t)
    else:
        data[:] = y.reshape(n_ep, n_t, n_sel).transpose(0, 2, 1)
    epochs._data[:, sel] = data
    return epochs


# ---------------------------------------------------------------------------
# tSSS / MaxFilter
# ---------------------------------------------------------------------------

#: Expansion origin / movement-compensation destination in head coordinates.
#: 4 cm above the fiducial midpoint is the usual stand-in when the recording
#: carries no digitised head shape to fit a sphere to.
MAXFILTER_ORIGIN = (0.0, 0.0, 0.04)


def cola_st_duration(sfreq, target=10.0, span=1.0):
    """Nearest tSSS buffer duration whose sample count is COLA-valid.

    ``mne.preprocessing.maxwell_filter(..., st_overlap=True)`` (the default)
    runs the temporal projection through a 50 % overlapping Hann window and
    refuses to start unless that window satisfies the constant-overlap-add
    condition, which here means an even number of samples.  The CTF sampling
    rate of 585.9375 Hz puts the conventional 10 s buffer at 5859 samples --
    odd -- so ``st_duration=10.`` raises before any filtering happens.  Nudging
    the buffer by a couple of milliseconds fixes it.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the recording.
    target : float
        Buffer length to aim for, in seconds.  10 s is the MaxFilter default
        and the shortest buffer that is generally considered safe: shorter
        buffers raise the chance that brain signal correlates with the
        external subspace and gets projected out.
    span : float
        Half-width of the search window around `target`, in seconds.

    Returns
    -------
    st_duration : float
        Buffer length in seconds, within `span` of `target`.
    """
    from scipy.signal import check_COLA, windows

    sfreq = float(sfreq)
    lo = max(2, int(np.floor((target - span) * sfreq)))
    hi = int(np.ceil((target + span) * sfreq))
    best = None
    for n in range(lo, hi + 1):
        if n % 2:  # hann + 50 % overlap needs n // 2 == n / 2
            continue
        if not check_COLA(windows.hann(n, sym=False), n, n // 2):
            continue
        if best is None or abs(n - target * sfreq) < abs(best - target * sfreq):
            best = n
    if best is None:
        raise ValueError(
            f"no COLA-valid tSSS buffer within {span} s of {target} s at {sfreq} Hz"
        )
    return best / sfreq


def maxwell_origin(info, fallback=MAXFILTER_ORIGIN, verbose=False):
    """Sphere origin for the SSS expansion, in head coordinates.

    Falls back to `fallback` when the recording has no digitised head shape --
    CTF datasets saved without a Polhemus scan only carry the three fiducials,
    and :func:`mne.bem.fit_sphere_to_headshape` needs at least four points.
    """
    from mne.bem import fit_sphere_to_headshape

    try:
        _, origin, _ = fit_sphere_to_headshape(info, units="m", verbose=verbose)
    except Exception:
        origin = fallback
    return tuple(float(i) for i in origin)


def ctf_head_positions(raw, decim_hz=0.2, verbose=False):
    """Continuous head position from the CTF HLC channels, for movement comp.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        CTF recording with continuous head localisation enabled.
    decim_hz : float | None
        Rate to thin the position estimates down to before they are handed to
        the Maxwell filter.  The SSS basis is rebuilt and re-regularised once
        per distinct position, so the native ~19 Hz CTF rate makes movement
        compensation roughly two orders of magnitude more expensive than the
        rest of the filter.  Heads drift over tens of seconds, not tens of
        milliseconds, so one update every few seconds captures essentially all
        of the real motion.  ``None`` keeps every estimate.

    Returns
    -------
    head_pos : array, shape (n_pos, 10) | None
        In the format :func:`mne.preprocessing.maxwell_filter` expects, or
        None when the recording has no HLC channels.
    """
    from mne.chpi import compute_head_pos, extract_chpi_locs_ctf

    if not any(ch.startswith("HLC") for ch in raw.ch_names):
        return None
    chpi_locs = extract_chpi_locs_ctf(raw, verbose=verbose)
    head_pos = compute_head_pos(raw.info, chpi_locs, verbose=verbose)
    if decim_hz is not None and len(head_pos) > 1:
        rate = 1.0 / np.median(np.diff(head_pos[:, 0]))
        step = max(1, int(round(rate / float(decim_hz))))
        if step > 1:
            head_pos = head_pos[::step]
    return head_pos


def head_pos_summary(head_pos):
    """Translation (mm) and rotation (deg) excursion of a `head_pos` array.

    Both are measured against the median position of the recording, so these
    describe how much the head moved *within* the block -- not how far it sat
    from the movement-compensation destination.
    """
    from mne.chpi import head_pos_to_trans_rot_t

    head_pos = np.asarray(head_pos, dtype=np.float64)
    trans, rot, _ = head_pos_to_trans_rot_t(head_pos)
    dist_mm = 1e3 * np.linalg.norm(trans - np.median(trans, axis=0), axis=1)
    ref = rot[len(rot) // 2]
    cos = (np.trace(ref.T @ rot, axis1=1, axis2=2) - 1.0) / 2.0
    ang_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return {
        "n_positions": int(len(head_pos)),
        "translation_mm_rms": float(np.sqrt(np.mean(dist_mm**2))),
        "translation_mm_max": float(dist_mm.max()),
        "rotation_deg_rms": float(np.sqrt(np.mean(ang_deg**2))),
        "rotation_deg_max": float(ang_deg.max()),
    }


def maxwell_bad_channels(
    raw,
    origin,
    coord_frame="head",
    int_order=8,
    ext_order=3,
    ignore_ref=False,
    tmax=120.0,
    limit=7.0,
    duration=5.0,
    min_count=5,
    verbose=False,
):
    """Noisy/flat sensors as seen by the SSS expansion.

    Marking bad channels before Maxwell filtering is not optional: the
    expansion is fitted to *all* sensors at once, so one dead or noisy sensor
    left in is smeared over the whole array by the reconstruction.

    `tmax` crops a copy of the recording first, because the detector Maxwell
    filters every `duration`-second chunk it looks at and that cost is linear
    in the length of the block.
    """
    from mne.preprocessing import find_bad_channels_maxwell

    sub = raw.copy().crop(tmax=min(tmax, raw.times[-1])) if tmax else raw
    noisy, flat = find_bad_channels_maxwell(
        sub,
        origin=origin,
        coord_frame=coord_frame,
        int_order=int_order,
        ext_order=ext_order,
        ignore_ref=ignore_ref,
        limit=limit,
        duration=duration,
        min_count=min_count,
        bad_condition="warning",
        verbose=verbose,
    )
    return list(noisy), list(flat)


def apply_tsss(
    raw,
    origin=None,
    destination=MAXFILTER_ORIGIN,
    coord_frame="head",
    int_order=8,
    ext_order=3,
    st_duration=10.0,
    st_correlation=0.98,
    movement_compensation=True,
    head_pos_decim_hz=0.2,
    detect_bad_channels=True,
    bad_channel_tmax=120.0,
    ignore_ref=False,
    regularize="in",
    bad_condition="warning",
    empty_room=False,
    verbose=False,
):
    """tSSS with movement compensation for a CTF recording.

    Wraps :func:`mne.preprocessing.maxwell_filter` with the pieces this dataset
    needs: an origin that survives the missing head shape, a COLA-valid buffer
    length for the 585.9375 Hz sampling rate, head positions read off the HLC
    channels, and bad-channel detection beforehand.

    Must be run on continuous raw data, before epoching: the temporal
    projection needs long uninterrupted buffers, and movement compensation
    needs the continuous head position.

    Parameters
    ----------
    raw : instance of mne.io.Raw
        Preloaded CTF recording.  Third-order gradient compensation may
        already be applied -- MNE folds the compensation matrix into the SSS
        basis as long as `ignore_ref` is False, which is why that is the
        default here.
    origin : array-like, shape (3,) | None
        Expansion origin.  None asks :func:`maxwell_origin`, which fits the
        head shape when there is one and falls back to `MAXFILTER_ORIGIN`.
    destination : array-like, shape (3,) | str | None
        Head position the data are re-rendered at.  Keeping this fixed across
        runs and subjects is what makes a given sensor mean approximately the
        same thing in a group sensor-level analysis; it buys comparability,
        not anatomy.  None leaves every block at its own head position.
    st_duration : float | None
        Target tSSS buffer length in seconds, snapped to the nearest
        COLA-valid value by :func:`cola_st_duration`.  None gives plain SSS.
    movement_compensation : bool
        Re-render each sample at `destination` using the measured head
        position.  Silently falls back to off when there are no HLC channels.
    empty_room : bool
        Switch to device coordinates with no destination and no movement
        compensation, for recordings with no head in the dewar.  Note that an
        empty-room file filtered this way is *not* in the same SSS basis as
        the task blocks; for a noise covariance to match, re-run it with
        :func:`mne.preprocessing.maxwell_filter_prepare_emptyroom` against the
        task block it is meant to go with.

    Returns
    -------
    raw_sss : instance of mne.io.Raw
        Maxwell-filtered copy.  Non-MEG channels (stim, EOG, HLC, ADC) pass
        through untouched.
    report : dict
        What was actually applied, for logging next to the output file.
    """
    from mne.preprocessing import maxwell_filter

    if empty_room:
        coord_frame, destination, movement_compensation = "meg", None, False
        origin = (0.0, 0.0, 0.0) if origin is None else origin
    elif origin is None:
        origin = maxwell_origin(raw.info, verbose=verbose)
    origin = tuple(float(i) for i in origin)

    if st_duration is not None:
        st_duration = cola_st_duration(raw.info["sfreq"], target=st_duration)

    head_pos = None
    if movement_compensation:
        head_pos = ctf_head_positions(raw, decim_hz=head_pos_decim_hz, verbose=verbose)

    noisy, flat = [], []
    if detect_bad_channels:
        noisy, flat = maxwell_bad_channels(
            raw,
            origin=origin,
            coord_frame=coord_frame,
            int_order=int_order,
            ext_order=ext_order,
            ignore_ref=ignore_ref,
            tmax=bad_channel_tmax,
            verbose=verbose,
        )
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(noisy) | set(flat))

    report = {
        "origin": list(origin),
        "coord_frame": coord_frame,
        "destination": (
            None
            if destination is None
            else [float(i) for i in np.ravel(destination)]
        ),
        "int_order": int(int_order),
        "ext_order": int(ext_order),
        "st_duration": None if st_duration is None else float(st_duration),
        "st_correlation": float(st_correlation),
        "compensation_grade": int(raw.compensation_grade or 0),
        "ignore_ref": bool(ignore_ref),
        "bads_noisy": list(noisy),
        "bads_flat": list(flat),
        "bads_applied": list(raw.info["bads"]),
        "movement_compensation": head_pos is not None,
    }
    if head_pos is not None:
        report["head_motion"] = head_pos_summary(head_pos)

    raw_sss = maxwell_filter(
        raw,
        origin=origin,
        coord_frame=coord_frame,
        int_order=int_order,
        ext_order=ext_order,
        st_duration=st_duration,
        st_correlation=st_correlation,
        head_pos=head_pos,
        destination=destination,
        ignore_ref=ignore_ref,
        regularize=regularize,
        bad_condition=bad_condition,
        verbose=verbose,
    )
    return raw_sss, report



if __name__ == "__main__":
    import time

    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    n = 1_000_000

    # reference head in CTF-ish dewar coordinates (metres)
    ref = np.array([[0.10, 0.0, 0.05], [0.0, 0.075, 0.05], [0.0, -0.075, 0.05]])

    # ground-truth motion: slow drift + wobble
    t = np.linspace(0, 1, n)
    true_rv = np.c_[0.02 * np.sin(2 * np.pi * t), 0.01 * t, -0.015 * np.cos(np.pi * t)]
    true_tr = np.c_[0.003 * t, 0.002 * np.sin(4 * np.pi * t), -0.001 * t]
    R = Rotation.from_rotvec(true_rv)
    pts = np.einsum("nij,kj->nki", R.as_matrix(), ref) + true_tr[:, None, :]

    nas, lpa, rpa = pts[:, 0], pts[:, 1], pts[:, 2]
    R_rel_true = R * R[0].inv()  # true motion relative to the first sample

    fiducials_to_6dof(nas[:10], lpa[:10], rpa[:10])  # warm up the jit
    for method in ("frame", "kabsch"):
        t0 = time.perf_counter()
        m = fiducials_to_6dof(nas, lpa, rpa, rotation="rotvec", degrees=False,
                              frame="device", method=method)
        dt = time.perf_counter() - t0
        rot_err = np.abs(m[:, 3:] - R_rel_true.as_rotvec()).max()
        print(f"{method:7s} {n:,} samples in {dt * 1000:7.1f} ms  "
              f"max rotvec error {rot_err:.2e} rad")

    # Euler angles must match scipy's intrinsic XYZ convention
    m = fiducials_to_6dof(nas[:1000], lpa[:1000], rpa[:1000], degrees=True,
                          frame="device")
    R_rel = Rotation.from_rotvec(true_rv[:1000]) * Rotation.from_rotvec(true_rv[0]).inv()
    print("max Euler mismatch vs scipy (deg):",
          np.abs(m[:, 3:] - R_rel.as_euler("XYZ", degrees=True)).max())
    # the head-frame origin is the LPA/RPA midpoint, which the rotation moves
    # too, so the expected device-frame translation is R_t @ mid + tr_t
    mid = 0.5 * (ref[1] + ref[2])
    o_true = np.einsum("nij,j->ni", R[:1000].as_matrix(), mid) + true_tr[:1000]
    print("translation check (mm):",
          np.abs((m[:, :3] - (o_true - o_true[0])) * 1000).max())
    summary = head_motion_summary(m * np.r_[[1000] * 3, [1] * 3])
    print({k: np.round(v, 4) for k, v in summary.items()}, "(mm / deg)")
