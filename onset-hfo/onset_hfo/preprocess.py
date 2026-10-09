"""Preprocessing: from a raw recording to the channels a detector sees.

Four steps, in this order, each of which is logged into ``Prepared.steps`` so
that a report can state exactly what was done to the signal:

1. **Channel selection** -- keep intracranial data channels (ECoG/SEEG/EEG),
   drop DC/trigger/ECG/misc channels and everything the dataset flagged ``bad``.
2. **High-pass** at 1 Hz to remove drift.
3. **Notch** at the mains frequency and its harmonics. Narrow notches
   (2 Hz wide) are used on purpose: harmonics at 180/240 Hz sit inside the
   ripple band, and a wide notch would carve a hole in the signal we are
   trying to measure.
4. **Bipolar montage** -- subtract neighbouring contacts on the same
   electrode. This is standard for HFO work: a common reference shares its
   noise with every channel and produces HFOs that appear everywhere at once.

Caveat worth knowing (and stated in the report): on a rectangular grid,
"neighbouring by number" (G8 - G9) is not always neighbouring in space,
because numbering wraps at the end of a row. For depth electrodes and strips
the numbering is spatially ordered, which is the case that matters most here.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from onset_hfo.config import BANDS, PreprocessConfig
from onset_hfo.datasets import Recording

__all__ = ["Prepared", "prepare", "ICA_METHODS", "ica_methods_available", "default_ica_method", "bipolar_pairs", "describe", "effective_reference",
           "filter_description", "learn_ptp_threshold", "REFERENCES", "laplacian_neighbours",
           "parse_grid_columns", "LAPLACIAN_STRIP_MAX"]

#: The reference schemes by name, with the sentence each gets in the steps.
REFERENCES = {
    "bipolar": "re-reference to neighbouring contacts (bipolar)",
    "average": "re-reference to the common average",
    "median": "re-reference to the common median",
    "shaft": "re-reference each contact to the mean of its own shaft",
    "laplacian": "re-reference each contact to the mean of its neighbours (Laplacian)",
    "none": "leave the recording's own reference",
}
FILTER_METHODS = ("fir", "iir")
FILTER_PHASES = ("zero", "minimum")

_CONTACT_RE = re.compile(r"^([A-Za-z]+[A-Za-z']*?)(\d{1,3})$")
#: Channel name prefixes that are never intracranial recordings in this dataset.
_NON_BRAIN = ("DC", "EKG", "ECG", "EMG", "EOG", "TRIG", "STIM", "REF", "EVENT", "MARK")


@dataclass
class Prepared:
    """Preprocessed signals plus the story of how they were produced."""

    data: np.ndarray              #: (n_channels, n_times), microvolts
    ch_names: list[str]           #: bipolar pair names ("G1-G2") or contact names
    sfreq: float
    t_offset: float               #: seconds; add to a local time to get file time
    montage: str                  #: "bipolar" or "monopolar"
    #: Mains frequency actually notched, resolved from the config or the
    #: recording. Carried here so no later step has to guess it again.
    line_freq: float = 60.0
    pairs: list[tuple[str, str]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    recording: Recording | None = None
    #: Seconds marked by the artifact annotators, each ``{"channel": name or
    #: None (every channel), "t_start", "t_stop" (file seconds), "reason"}``.
    #: The data-quality stage sets them aside; nothing here deletes them.
    annotations: list[dict] = field(default_factory=list)
    #: The ICA stage's record when it ran: the method, the components, their
    #: sources and loadings for the panel, MNE's scores, what was suggested
    #: and what the reviewer removed. ``None`` when the stage is off.
    ica: dict | None = None

    @property
    def duration(self) -> float:
        return self.data.shape[1] / self.sfreq

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    def contacts_of(self, ch: str) -> list[str]:
        """The physical contact(s) a (possibly bipolar) channel is built from."""
        if self.montage == "bipolar" and ch in self.ch_names:
            a, b = self.pairs[self.ch_names.index(ch)]
            return [a, b]
        return [ch]

    def index(self, ch: str) -> int:
        return self.ch_names.index(ch)

    def segment(self, ch: str, t0: float, t1: float, absolute: bool = True) -> np.ndarray:
        """Signal for one channel between two times (default: file times)."""
        if absolute:
            t0, t1 = t0 - self.t_offset, t1 - self.t_offset
        i0 = max(0, int(round(t0 * self.sfreq)))
        i1 = min(self.data.shape[1], int(round(t1 * self.sfreq)))
        return self.data[self.index(ch), i0:i1]


def _is_brain_channel(name: str, ch_type: str) -> bool:
    if ch_type not in ("ecog", "seeg", "eeg"):
        return False
    upper = name.upper()
    return not any(upper.startswith(p) for p in _NON_BRAIN)


def bipolar_pairs(ch_names: list[str], exclude: set[str] | None = None) -> list[tuple[str, str]]:
    """Consecutive contacts on the same electrode, e.g. ``("AD1", "AD2")``.

    Both contacts must be present and neither may be excluded, so a bad
    contact removes the two pairs it would have formed rather than silently
    bridging a gap (AD1-AD3 would mix two distances and is never produced).
    """
    exclude = exclude or set()
    by_lead: dict[str, dict[int, str]] = {}
    for name in ch_names:
        m = _CONTACT_RE.match(name)
        if not m:
            continue
        by_lead.setdefault(m.group(1), {})[int(m.group(2))] = name
    pairs: list[tuple[str, str]] = []
    for lead in sorted(by_lead):
        numbers = sorted(by_lead[lead])
        for n in numbers:
            if n + 1 in by_lead[lead]:
                a, b = by_lead[lead][n], by_lead[lead][n + 1]
                if a not in exclude and b not in exclude:
                    pairs.append((a, b))
    return pairs


#: ICA solvers MNE can run. Infomax is MNE's own; FastICA is scikit-learn's
#: and Picard is the `python-picard` package, so each of those is listed only
#: when its package imports.
ICA_METHODS = ("fastica", "infomax", "picard")
SOLVER_PACKAGES = {"fastica": "scikit-learn", "picard": "python-picard"}


def _importable(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def ica_methods_available() -> tuple[str, ...]:
    """The solvers that will run on this machine, in `ICA_METHODS` order."""
    have = {"fastica": _importable("sklearn"), "infomax": True, "picard": _importable("picard")}
    return tuple(m for m in ICA_METHODS if have[m])


def default_ica_method() -> str:
    """The first solver that runs here: ``fastica`` with scikit-learn
    installed, else ``infomax``, which needs nothing beyond MNE."""
    available = ica_methods_available()
    return "fastica" if "fastica" in available else available[0]


def effective_reference(cfg: PreprocessConfig) -> str:
    """The scheme in force: the named one, or what the two older flags say."""
    if cfg.reference:
        name = str(cfg.reference).lower()
        if name not in REFERENCES:
            raise ValueError(f"reference must be one of {', '.join(REFERENCES)}, "
                             f"not {cfg.reference!r}")
        return name
    return "bipolar" if cfg.bipolar else "average" if cfg.average_reference else "none"


def _shaft_of(name: str) -> str:
    m = _CONTACT_RE.match(name)
    return m.group(1).upper() if m else name.upper()


#: A lead with more contacts than this, no positions and no columns given is
#: not taken as a strip: 32 contacts in one line is almost always a grid, and
#: a grid read as a line makes the last contact of a row the neighbour of the
#: first of the next.
LAPLACIAN_STRIP_MAX = 16
#: Contacts this much farther than a contact's nearest neighbour on its lead
#: are not its neighbours: on a square grid the four sides are in and the
#: diagonals (1.41x) out.
NEIGHBOUR_REACH = 1.25


def parse_grid_columns(text: str) -> tuple[tuple[str, int], ...]:
    """``"G:8, LT:4"`` as `PreprocessConfig.grid_columns` wants it. A part that
    is not ``name:columns`` with a positive count raises ValueError."""
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        if not part.strip():
            continue
        name, sep, count = part.partition(":")
        if not sep or not name.strip() or not count.strip().isdigit() or int(count) < 1:
            raise ValueError(f"{part.strip()!r} is not a grid and its columns, as G:8")
        out.append((name.strip().upper(), int(count)))
    return tuple(out)


def laplacian_neighbours(names: list[str], positions: dict | None = None,
                         grid_columns=(), source: dict | None = None
                         ) -> tuple[dict[str, list[str]], list[str]]:
    """Each contact's neighbours on its own lead, and what was done per lead.

    Neighbours come from the contacts' positions when every contact of the
    lead has one (those within `NEIGHBOUR_REACH` of its nearest); else from
    the grid's columns when given; else, for a lead of up to
    `LAPLACIAN_STRIP_MAX` contacts, the contacts numbered either side. A
    contact with no neighbour present is absent from the result, and is left
    as recorded. Positions are matched to names without regard to case;
    `source` names where a contact's position came from, for the notes."""
    columns = {str(k).upper(): int(v) for k, v in (grid_columns or ())}
    leads: dict[str, dict[int, str]] = {}
    for name in names:
        m = _CONTACT_RE.match(name)
        if m:
            leads.setdefault(m.group(1).upper(), {})[int(m.group(2))] = name
    placed = {str(k).upper(): v for k, v in (positions or {}).items()}
    source = {str(k).upper(): v for k, v in (source or {}).items()}
    out: dict[str, list[str]] = {}
    notes: list[str] = []
    for lead in sorted(leads):
        numbered = leads[lead]
        members = [numbered[k] for k in sorted(numbered)]
        located = len(members) > 1 and all(
            n.upper() in placed and np.all(np.isfinite(placed[n.upper()])) for n in members)
        if located:
            xyz = np.array([placed[n.upper()] for n in members], dtype=float)
            dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            np.fill_diagonal(dist, np.inf)
            for i, name in enumerate(members):
                nearest = dist[i].min()
                if np.isfinite(nearest) and nearest > 0:
                    near = [members[j] for j in np.flatnonzero(dist[i] <= nearest * NEIGHBOUR_REACH)]
                    out[name] = near
            where = {source.get(n.upper(), "") for n in members} - {""}
            notes.append(f"{lead}: {len(members)} contacts, neighbours from their positions"
                         + (f" in {', '.join(sorted(where))}" if where else
                            " in the recording"))
        elif lead in columns:
            width = columns[lead]
            for number, name in numbered.items():
                row, col = divmod(number - 1, width)
                around = [(row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)]
                near = [numbered.get(r * width + c + 1) for r, c in around
                        if 0 <= c < width and r >= 0]
                near = [n for n in near if n is not None]
                if near:
                    out[name] = near
            rows = -(-max(numbered) // width)
            notes.append(f"{lead}: {rows}×{width} grid, its four neighbours")
        elif len(members) <= LAPLACIAN_STRIP_MAX:
            for number, name in numbered.items():
                near = [numbered[k] for k in (number - 1, number + 1) if k in numbered]
                if near:
                    out[name] = near
            notes.append(f"{lead}: {len(members)} in a line, the contacts either side")
        else:
            notes.append(f"{lead}: {len(members)} contacts left as recorded — give its "
                         f"columns ({lead}:8) if it is a grid")
    return out, notes


def _iir_params(cfg: PreprocessConfig) -> dict:
    return {"order": int(cfg.iir_order), "ftype": "butter", "output": "sos"}


def filter_description(cfg: PreprocessConfig, sfreq: float) -> str:
    """What the chosen design is, in MNE's own terms: the length of the FIR
    in samples and seconds, or the order of the Butterworth, with the
    transition band. Read off ``mne.filter.create_filter`` rather than
    restated, so the sentence cannot drift from the filter."""
    from mne.filter import create_filter

    if not (cfg.highpass or cfg.lowpass):
        return "no high- or low-pass"
    rate = float(cfg.resample) if cfg.resample else float(sfreq)
    kwargs = {}
    if cfg.transition_bandwidth:
        kwargs["l_trans_bandwidth"] = float(cfg.transition_bandwidth)
        kwargs["h_trans_bandwidth"] = float(cfg.transition_bandwidth)
    try:
        if cfg.filter_method == "iir":
            design = create_filter(None, rate, cfg.highpass or None, cfg.lowpass,
                                   method="iir", iir_params=_iir_params(cfg),
                                   phase="zero", verbose="ERROR")
            order = int(design.get("order", cfg.iir_order))
            return (f"Butterworth IIR of order {order}, applied forwards and backwards "
                    f"(zero phase, effective order {2 * order})")
        taps = create_filter(None, rate, cfg.highpass or None, cfg.lowpass, method="fir",
                             phase=cfg.filter_phase, fir_design="firwin", verbose="ERROR",
                             **kwargs)
        n = int(np.asarray(taps).shape[0])
        trans = (f"{float(cfg.transition_bandwidth):g} Hz" if cfg.transition_bandwidth
                 else "MNE's automatic")
        return (f"windowed FIR of {n} taps ({n / rate:.2f} s), "
                f"{'zero' if cfg.filter_phase == 'zero' else 'minimum'} phase, "
                f"{trans} transition band")
    except Exception as error:      # noqa: BLE001 - a description must not refuse
        return f"{cfg.filter_method} filter (could not be characterised: {error})"


def learn_ptp_threshold(epochs: np.ndarray, n_folds: int = 5,
                        n_candidates: int = 25) -> float:
    """A peak-to-peak ceiling for one channel, learned from its own epochs by
    cross-validation, the way ``autoreject`` learns its global threshold.

    For each candidate ceiling, the epochs under it in the training folds
    are averaged and compared with the median of the held-out fold; the
    ceiling with the smallest error wins. Said plainly: it keeps the epochs
    that look like the typical epoch, and a discharge does not. That is the
    hazard :class:`onset_hfo.config.QualityConfig` documents, and why this is
    an option rather than a default.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 2 or epochs.shape[0] < 2 * n_folds:
        return float("inf")
    ptp = epochs.max(axis=1) - epochs.min(axis=1)
    candidates = np.unique(np.quantile(ptp, np.linspace(0.3, 1.0, n_candidates)))
    n = epochs.shape[0]
    folds = np.array_split(np.arange(n), n_folds)
    best, best_error = float(candidates[-1]), float("inf")
    for ceiling in candidates:
        errors = []
        for held in folds:
            train = np.setdiff1d(np.arange(n), held)
            kept = train[ptp[train] <= ceiling]
            if kept.size == 0:
                errors.append(float("inf"))
                continue
            mean = epochs[kept].mean(axis=0)
            target = np.median(epochs[held], axis=0)
            errors.append(float(np.sqrt(np.mean((mean - target) ** 2))))
        error = float(np.mean(errors))
        if error < best_error:
            best, best_error = float(ceiling), error
    return best


def _annotate(raw, cfg: PreprocessConfig, t_offset: float, steps: list[str]) -> list[dict]:
    """Run the artifact annotators MNE offers on the filtered, monopolar
    signal, and return the seconds they marked in file time."""

    found: list[dict] = []
    if cfg.annotate_muscle:
        from mne.preprocessing import annotate_muscle_zscore

        # MNE's muscle detector knows scalp types; an intracranial contact is
        # the same arithmetic. Done on a copy so the recording keeps its types.
        probe = raw.copy()
        probe.set_channel_types({name: "eeg" for name in probe.ch_names}, verbose="ERROR")
        nyquist = float(probe.info["sfreq"]) / 2.0
        band = (110.0, min(140.0, 0.9 * nyquist))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            notes, _scores = annotate_muscle_zscore(
                probe, ch_type="eeg", threshold=float(cfg.muscle_z),
                min_length_good=0.1, filter_freq=band, verbose="ERROR")
        for onset, duration in zip(notes.onset, notes.duration, strict=True):
            found.append({"channel": None, "t_start": float(onset) + t_offset,
                          "t_stop": float(onset + duration) + t_offset,
                          "reason": "annotated_muscle"})
        seconds = float(sum(notes.duration))
        steps.append(f"marked {seconds:.1f} s of broadband high-frequency bursts across "
                     f"the montage as muscle or movement (z > {cfg.muscle_z:g} in "
                     f"{band[0]:.0f}–{band[1]:.0f} Hz); set aside by the quality stage")
    if cfg.annotate_amplitude:
        from mne.preprocessing import annotate_amplitude

        data = raw.get_data() * 1e6
        sfreq = float(raw.info["sfreq"])
        width = max(1, int(round(sfreq)))
        if cfg.amplitude_ptp_uv is not None:
            ceilings = {name: float(cfg.amplitude_ptp_uv) for name in raw.ch_names}
            how = f"a {float(cfg.amplitude_ptp_uv):g} µV peak-to-peak ceiling"
        else:
            ceilings = {}
            for i, name in enumerate(raw.ch_names):
                n = data.shape[1] // width
                epochs = data[i, :n * width].reshape(n, width) if n else data[i:i + 1]
                ceilings[name] = learn_ptp_threshold(epochs)
            finite = [c for c in ceilings.values() if np.isfinite(c)]
            how = (f"a peak-to-peak ceiling learned per contact by cross-validation "
                   f"(median {np.median(finite):.0f} µV)" if finite else
                   "a learned ceiling (too little signal to learn one)")
        marked = 0.0
        for name, ceiling in ceilings.items():
            if not np.isfinite(ceiling):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                notes, _bads = annotate_amplitude(
                    raw, peak=float(ceiling) * 1e-6, flat=None, bad_percent=100,
                    min_duration=0.005, picks=[name], verbose="ERROR")
            for onset, duration in zip(notes.onset, notes.duration, strict=True):
                found.append({"channel": name, "t_start": float(onset) + t_offset,
                              "t_stop": float(onset + duration) + t_offset,
                              "reason": "annotated_amplitude"})
                marked += float(duration)
        steps.append(f"marked {marked:.1f} contact-seconds above {how}; set aside by "
                     f"the quality stage, never interpolated")
    return found


def _regress_out(raw, picks: list[str], artifact: list[str]) -> np.ndarray:
    """Ordinary least squares of each picked channel on the artifact channels,
    the residual written back: the arithmetic of MNE's ``regress_artifact``.

    Done here rather than through MNE's function because that one refuses
    intracranial channels unless an average-reference projector has been
    added to the recording, which this pipeline has no business doing. The
    fit is on the filtered signal, means removed, so a DC offset on the
    lead is not regressed into every channel. Returns the coefficients,
    shaped (picked channels, artifact channels).
    """
    data = raw.get_data(picks=picks)
    lead = raw.get_data(picks=artifact)
    design = (lead - lead.mean(axis=1, keepdims=True)).T          # (times, artifacts)
    target = (data - data.mean(axis=1, keepdims=True)).T          # (times, channels)
    betas, _res, _rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    cleaned = (target - design @ betas).T + data.mean(axis=1, keepdims=True)
    index = [raw.ch_names.index(name) for name in picks]
    raw._data[index, :] = cleaned
    return betas.T


def _has_positions(raw) -> bool:
    """Whether every channel carries a finite, non-zero position."""
    locs = np.array([ch["loc"][:3] for ch in raw.info["chs"]], dtype=float)
    return bool(locs.size) and bool(np.isfinite(locs).all()) and bool(np.any(locs != 0))


def _ica_ecg_channel(rec: Recording) -> str | None:
    """An ECG lead in the recording, by type or by name, if it has one."""
    raw = rec.raw
    for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=False):
        if kind == "ecg" or name.upper().startswith(("ECG", "EKG")):
            return name
    return None


def _fit_ica(raw, cfg: PreprocessConfig, rec: Recording, steps: list[str]) -> dict:
    """Fit MNE's ICA on the filtered monopolar brain channels, score the
    components the way MNE does for muscle and ECG, apply only the reviewer's
    chosen exclusions, and return the record the Components panel reads.

    The sources and loadings are kept in the record so the panel can draw a
    component without the fitted object, which is not serialisable.
    """
    import inspect

    from mne.preprocessing import ICA

    n_channels = len(raw.ch_names)
    wanted = (int(cfg.ica_n_components) if cfg.ica_n_components
              else min(20, n_channels - 1))
    n_components = max(1, min(wanted, n_channels - 1))
    seed = int(cfg.ica_seed)
    kwargs = ({"rng": seed} if "rng" in inspect.signature(ICA.__init__).parameters
              else {"random_state": seed})
    ica = ICA(n_components=n_components, method=cfg.ica_method, max_iter="auto", **kwargs)
    notes: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw, verbose="ERROR")
        fitted = int(ica.n_components_)
        sources = ica.get_sources(raw).get_data()
        loadings = np.asarray(ica.get_components(), dtype=float)       # (channels, components)
        variance = np.asarray(ica.pca_explained_variance_, dtype=float)
        share = (variance[:fitted] / variance.sum()).tolist() if variance.sum() > 0 else []
        # MNE's muscle score has a spatial term that needs electrode positions;
        # without them every component scores 0, which would read as "clean"
        # when it means "not scored". So it is not run, and the record says so.
        muscle_idx, muscle_scores = [], []
        muscle_scored = False
        if _has_positions(raw):
            try:
                muscle_idx, muscle_scores = ica.find_bads_muscle(raw, verbose="ERROR")
                muscle_scored = True
            except Exception as error:      # noqa: BLE001 - said, not raised
                notes.append(f"MNE's muscle scoring did not run here ({type(error).__name__}).")
        else:
            notes.append("Not scored for muscle: MNE's muscle score needs electrode positions "
                         "and this recording carries none. The share of power above 40 Hz "
                         "is the guide instead.")
        ecg_name = _ica_ecg_channel(rec)
        ecg_idx, ecg_scores = [], []
        if ecg_name is not None:
            try:
                probe = raw.copy()
                lead = rec.raw.copy().pick([ecg_name])
                if abs(lead.info["sfreq"] - probe.info["sfreq"]) > 1e-9:
                    lead.resample(probe.info["sfreq"], verbose="ERROR")
                lead.crop(tmax=probe.times[-1], verbose="ERROR")
                probe.add_channels([lead], force_update_info=True)
                ecg_idx, ecg_scores = ica.find_bads_ecg(probe, ch_name=ecg_name,
                                                        method="correlation",
                                                        threshold="auto", verbose="ERROR")
            except Exception as error:      # noqa: BLE001 - said, not raised
                notes.append(f"ECG scoring against {ecg_name} did not run "
                             f"({type(error).__name__}).")
        else:
            notes.append("No ECG lead in this recording, so no component is scored for ECG.")
    # The reviewer's choice, and only that, is applied.
    exclude = sorted({int(i) for i in cfg.ica_exclude if 0 <= int(i) < fitted})
    ignored = sorted({int(i) for i in cfg.ica_exclude} - set(exclude))
    if ignored:
        notes.append(f"Component index(es) {', '.join(str(i) for i in ignored)} do not exist "
                     f"in this fit of {fitted} and were ignored.")
    if exclude:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica.apply(raw, exclude=exclude, verbose="ERROR")
    # A spectral share per component: the fraction of its power above 40 Hz,
    # for the panel's table, beside MNE's scores. A guide to the eye only.
    hf_share = _high_frequency_share(sources, float(raw.info["sfreq"]))
    steps.append(
        f"ICA ({cfg.ica_method}, {fitted} components, seed {seed}) on {n_channels} channels: "
        + (f"MNE suggests muscle {', '.join(str(i) for i in muscle_idx)}" if len(muscle_idx)
           else ("MNE suggests no muscle component" if muscle_scored
                 else "not scored for muscle (no electrode positions)"))
        + (f"; ECG {', '.join(str(i) for i in ecg_idx)}" if len(ecg_idx)
           else ("; no ECG component" if ecg_name else "; no ECG lead to score against"))
        + (f"; removed {', '.join(str(i) for i in exclude)} (the reviewer's choice)"
           if exclude else "; nothing removed")
        + ". ICA can remove real HFO energy with the artefact; experimental.")
    return {
        "method": cfg.ica_method, "n_components": fitted, "seed": seed,
        "channels": list(raw.ch_names), "sfreq": float(raw.info["sfreq"]),
        "sources": sources.astype(np.float32), "loadings": loadings,
        "variance_share": share, "hf_share": hf_share,
        "muscle_scored": muscle_scored,
        "muscle_scores": [float(v) for v in np.asarray(muscle_scores).reshape(-1)],
        "ecg_scores": [float(v) for v in np.asarray(ecg_scores).reshape(-1)],
        "suggested_muscle": [int(i) for i in muscle_idx],
        "suggested_ecg": [int(i) for i in ecg_idx],
        "ecg_channel": ecg_name, "excluded": exclude, "notes": notes,
    }


def _high_frequency_share(sources: np.ndarray, sfreq: float, above_hz: float = 40.0) -> list[float]:
    if sources.size == 0:
        return []
    n = int(min(sources.shape[1], 4096))
    spectrum = np.abs(np.fft.rfft(sources[:, :n] - sources[:, :n].mean(axis=1, keepdims=True),
                                  axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    total = spectrum[:, freqs > 0.5].sum(axis=1)
    high = spectrum[:, freqs >= above_hz].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, high / total, 0.0)
    return [float(v) for v in share]


def _check(cfg: PreprocessConfig, sfreq: float) -> None:
    """Refuse settings that would produce numbers rather than a measurement.

    Every one of these runs perfectly well and returns a plausible-looking
    answer, which is exactly why they are refused here instead of being left to
    whoever reads the result. A low-pass under the high-pass passes nothing; a
    low-pass above the new Nyquist after downsampling is asking for a band the
    signal can no longer carry.
    """
    if cfg.highpass and cfg.lowpass and cfg.lowpass <= cfg.highpass:
        raise ValueError(
            f"low-pass {cfg.lowpass:g} Hz is at or below the high-pass "
            f"{cfg.highpass:g} Hz, which passes nothing")
    # `is not None` rather than truthiness: a resample of 0 Hz is nonsense, and
    # treating it as "no resampling" would accept it silently.
    if cfg.resample is not None and float(cfg.resample) <= 0:
        raise ValueError(
            f"resample rate must be positive, not {float(cfg.resample):g} Hz")
    rate = float(cfg.resample) if cfg.resample else float(sfreq)
    if cfg.lowpass and cfg.lowpass >= rate / 2.0:
        raise ValueError(
            f"low-pass {cfg.lowpass:g} Hz is at or above the Nyquist frequency "
            f"of {rate / 2.0:g} Hz"
            + (f" that resampling to {rate:g} Hz would leave" if cfg.resample
               else ""))
    if cfg.notch_width <= 0:
        raise ValueError(f"notch width must be positive, not {cfg.notch_width:g} Hz")
    if cfg.filter_method not in FILTER_METHODS:
        raise ValueError(f"filter_method must be one of {FILTER_METHODS}, "
                         f"not {cfg.filter_method!r}")
    if cfg.filter_phase not in FILTER_PHASES:
        raise ValueError(f"filter_phase must be one of {FILTER_PHASES}, "
                         f"not {cfg.filter_phase!r}")
    if cfg.filter_method == "iir" and not 1 <= int(cfg.iir_order) <= 16:
        raise ValueError(f"iir_order must be between 1 and 16, not {cfg.iir_order}")
    if cfg.transition_bandwidth is not None and float(cfg.transition_bandwidth) <= 0:
        raise ValueError("transition_bandwidth must be positive")
    if (cfg.transition_bandwidth and cfg.highpass
            and float(cfg.transition_bandwidth) >= float(cfg.highpass)):
        raise ValueError(
            f"a {float(cfg.transition_bandwidth):g} Hz transition band is as wide as the "
            f"{cfg.highpass:g} Hz high-pass, so the stop band would start below 0 Hz; "
            "it must be narrower than the cut-off")
    effective_reference(cfg)
    for entry in cfg.grid_columns or ():
        if (len(entry) != 2 or not str(entry[0]).strip()
                or int(entry[1]) != entry[1] or int(entry[1]) < 1):
            raise ValueError(f"grid_columns entries are (grid name, columns), not {entry!r}")
    if cfg.annotate_muscle and float(cfg.muscle_z) <= 0:
        raise ValueError("muscle_z must be positive")
    if cfg.annotate_amplitude and cfg.amplitude_ptp_uv is not None \
            and float(cfg.amplitude_ptp_uv) <= 0:
        raise ValueError("amplitude_ptp_uv must be positive")
    if cfg.ica:
        if cfg.ica_method not in ICA_METHODS:
            raise ValueError(f"ica_method must be one of {ICA_METHODS}, not {cfg.ica_method!r}")
        if cfg.ica_method not in ica_methods_available():
            package = SOLVER_PACKAGES.get(cfg.ica_method, cfg.ica_method)
            raise ValueError(
                f"the {cfg.ica_method} ICA solver needs the {package} package, which is not "
                f"installed on this machine; install it (pip install {package}) or choose "
                f"{' or '.join(ica_methods_available())}")
        if cfg.ica_n_components is not None and int(cfg.ica_n_components) < 1:
            raise ValueError("ica_n_components must be at least 1")
        if any(int(i) < 0 for i in cfg.ica_exclude):
            raise ValueError("ica_exclude indices must be non-negative")

def describe(cfg: PreprocessConfig, band: tuple[float, float],
             sfreq: float) -> tuple[str, list[str]]:
    """Plain English for a config, plus everything questionable about it.

    Lives here rather than in the panel that shows it, next to the `_check`
    whose refusals it mirrors. That is structural: a reviewer should learn that
    a 150 Hz low-pass is wrong from the red text under the control rather than
    from a dialog a minute into a re-analysis, and the only way the two stay
    in step is for them to be read and edited together.

    Returns the sentence and the list of concerns. The concerns that `_check`
    *raises* on are included, so a caller can tell "do not let this be applied"
    from "the reviewer should know".
    """
    rate = float(cfg.resample) if cfg.resample else float(sfreq)
    lines = []
    if cfg.highpass:
        lines.append(f"high-pass at {cfg.highpass:g} Hz")
    if cfg.lowpass:
        lines.append(f"low-pass at {cfg.lowpass:g} Hz")
    if cfg.notch:
        mains = f"{cfg.line_freq:g} Hz" if cfg.line_freq else "the mains frequency"
        lines.append(f"notch {mains}"
                     + (" and its harmonics" if cfg.notch_harmonics else " only")
                     + f", {cfg.notch_width:g} Hz wide")
    if cfg.highpass or cfg.lowpass:
        lines.append(f"as a {filter_description(cfg, sfreq)}")
    if cfg.resample:
        lines.append(f"resample to {cfg.resample:g} Hz")
    try:
        scheme = effective_reference(cfg)
    except ValueError:
        scheme = "none"
    lines.append(REFERENCES[scheme]
                 + (" on grid " + ", ".join(f"{g} ({c} columns)" for g, c in cfg.grid_columns)
                    if scheme == "laplacian" and cfg.grid_columns else ""))
    if cfg.annotate_muscle:
        lines.append(f"mark muscle and movement bursts (z > {cfg.muscle_z:g})")
    if cfg.annotate_amplitude:
        lines.append("mark seconds above "
                     + (f"{float(cfg.amplitude_ptp_uv):g} µV peak-to-peak"
                        if cfg.amplitude_ptp_uv is not None
                        else "a peak-to-peak ceiling learned per contact"))
    if cfg.regress_channels:
        lines.append(f"regress {', '.join(cfg.regress_channels)} out of every brain channel")
    if cfg.ica:
        how = (f"{int(cfg.ica_n_components)} components" if cfg.ica_n_components
               else "up to 20 components")
        lines.append(f"fit ICA ({cfg.ica_method}, {how})"
                     + (f" and remove component(s) {', '.join(str(i) for i in cfg.ica_exclude)}"
                        if cfg.ica_exclude else ", removing nothing until you choose"))
    if cfg.drop_bads:
        lines.append("drop channels the dataset flagged bad")
    if cfg.exclude:
        lines.append(f"drop {len(cfg.exclude)} channel(s) you marked")

    warnings = []
    if cfg.highpass and cfg.lowpass and cfg.lowpass <= cfg.highpass:
        warnings.append(f"The low-pass ({cfg.lowpass:g} Hz) is at or below the "
                        f"high-pass ({cfg.highpass:g} Hz). That passes nothing.")
    if cfg.lowpass and cfg.lowpass >= rate / 2:
        warnings.append(f"The low-pass ({cfg.lowpass:g} Hz) is at or above the "
                        f"Nyquist frequency of {rate / 2:g} Hz.")
    elif cfg.lowpass and cfg.lowpass < band[1]:
        warnings.append(f"The low-pass ({cfg.lowpass:g} Hz) cuts into the band "
                        f"being analysed ({band[0]:.0f}–{band[1]:.0f} Hz). The "
                        f"detector would still run and its rates would mean "
                        f"nothing.")
    if not BANDS.usable(rate, band):
        warnings.append(f"At {rate:g} Hz this recording cannot carry the "
                        f"{band[1]:.0f} Hz top of the band being analysed; it "
                        f"needs more than {2 * band[1]:.0f} Hz.")
    if cfg.highpass and cfg.highpass > band[0]:
        warnings.append(f"The high-pass ({cfg.highpass:g} Hz) is inside the band "
                        f"being analysed, which starts at {band[0]:.0f} Hz.")
    if not cfg.notch:
        warnings.append("With the notch off, mains harmonics sit inside the HFO "
                        "bands and are detected as oscillations.")
    if cfg.notch_width <= 0:
        warnings.append(f"A notch width of {cfg.notch_width:g} Hz is not a "
                        f"filter. It must be positive.")
    if cfg.resample is not None and float(cfg.resample) <= 0:
        warnings.append(f"A sampling rate of {float(cfg.resample):g} Hz is not "
                        f"a rate. It must be positive.")
    if cfg.notch_width > 4.0:
        warnings.append(f"A {cfg.notch_width:g} Hz notch is wide; its harmonics "
                        f"carve visible holes in the band being analysed.")
    if scheme == "none":
        warnings.append("Without re-referencing, a shared reference puts the "
                        "same noise on every channel, which reads as HFOs "
                        "appearing everywhere at once.")
    if scheme == "laplacian":
        warnings.append("The Laplacian takes each contact's neighbours from its name and "
                        "number, or its position: a contact quality marks bad still "
                        "feeds the contacts beside it.")
    if scheme in ("average", "median"):
        warnings.append("A common reference shares every channel's noise with "
                        "every other, which is the effect the bipolar montage "
                        "exists to avoid for HFO work.")
    if cfg.filter_method not in FILTER_METHODS or cfg.filter_phase not in FILTER_PHASES:
        warnings.append("The filter design must be FIR or IIR, zero or minimum phase.")
    if cfg.filter_method == "iir" and int(cfg.iir_order) > 8:
        warnings.append(f"A Butterworth of order {int(cfg.iir_order)} (effective "
                        f"{2 * int(cfg.iir_order)}) rings hard at a sharp discharge; "
                        "the ringing looks like a ripple.")
    if (cfg.transition_bandwidth and cfg.highpass
            and float(cfg.transition_bandwidth) >= float(cfg.highpass)):
        warnings.append(f"The {float(cfg.transition_bandwidth):g} Hz transition band is "
                        f"as wide as the {cfg.highpass:g} Hz high-pass; the stop band "
                        "would start below 0 Hz. It must be narrower than the cut-off.")
    if cfg.filter_phase == "minimum":
        warnings.append("A minimum-phase filter delays the signal, so event times "
                        "are late by part of the filter's length; the archive's "
                        "markings are not.")
    if cfg.annotate_muscle:
        warnings.append("The muscle band (110–140 Hz) lies inside the ripple band. "
                        "The detector marks seconds where it rises across the whole "
                        "montage at once, which a ripple on one contact does not do; "
                        "a widespread burst of real activity would still be marked.")
    if cfg.annotate_amplitude:
        warnings.append("An amplitude ceiling removes the loudest seconds, and on an "
                        "epileptic contact the loudest seconds are the discharges. "
                        "This project measured that and set its own segment test on "
                        "discontinuity instead; use this knowing what it removes.")
    if (cfg.annotate_muscle or cfg.annotate_amplitude):
        warnings.append("Marked seconds take effect only with 'Check data quality' on.")
    if cfg.regress_channels:
        warnings.append("Regression removes whatever part of each brain channel follows "
                        "the named channel, by least squares; a brain channel that "
                        "genuinely shares a rhythm with it loses that rhythm too.")
    if cfg.ica:
        warnings.append("ICA is experimental here: it can take real HFO energy out with "
                        "the artefact, and the literature is split on using it for HFO "
                        "work. Nothing is removed until you choose components on the "
                        "Components panel, and the report names what was removed.")
        if cfg.ica_method not in ica_methods_available():
            warnings.append(f"The {cfg.ica_method} solver is not installed; choose another.")
    return "; ".join(lines) + ".", warnings

def prepare(rec: Recording, cfg: PreprocessConfig | None = None, verbose: bool = True,
            positions: dict | None = None, positions_from: str = "") -> Prepared:
    """Run the four preprocessing steps and return the array a detector reads.

    The returned data is in **microvolts**, because every threshold and plot in
    this project is expressed in µV and silent unit changes are how analyses go
    wrong.

    `positions` (contact name -> x, y, z, any one unit) are contact positions
    from outside the recording -- a reader's electrode file -- for the
    Laplacian's neighbours; `positions_from` names where they came from for
    the steps. Positions the recording carries itself are used when these do
    not cover a lead. Nothing else reads them.
    """
    cfg = cfg or PreprocessConfig()
    _check(cfg, float(rec.raw.info["sfreq"]))
    raw = rec.raw.copy()
    steps: list[str] = []

    # 1. channel selection ------------------------------------------------
    types = raw.get_channel_types()
    keep = [n for n, t in zip(raw.ch_names, types, strict=False) if _is_brain_channel(n, t)]
    dropped_type = [n for n in raw.ch_names if n not in keep]
    if not keep:
        raise ValueError("No intracranial data channels found in this recording")
    # Channels to regress out ride along through the filters, so they are
    # compared with the brain channels on equal terms, and leave before the
    # montage. A name the recording does not carry is refused, not skipped.
    regress = list(dict.fromkeys(cfg.regress_channels))
    missing = [c for c in regress if c not in raw.ch_names]
    if missing:
        raise ValueError(f"no channel {', '.join(missing)} in this recording to regress out")
    regress = [c for c in regress if c not in keep]
    raw.pick(keep + regress)
    steps.append(f"kept {len(keep)} intracranial channels; dropped "
                 f"{len(dropped_type) - len(regress)} non-brain channels (DC/trigger/ECG/misc)"
                 + (f"; kept {', '.join(regress)} to regress out" if regress else ""))

    bads = [b for b in rec.bads if b in raw.ch_names]
    if cfg.drop_bads and bads:
        raw.drop_channels(bads)
        steps.append(f"dropped {len(bads)} channels flagged bad by the dataset: {', '.join(sorted(bads))}")

    # Channels the reviewer excluded themselves. Logged separately from the
    # dataset's own flags because they are a different kind of claim: one is
    # the archive's, the other is this reviewer's judgement on this window, and
    # a report that merged them would attribute the second to the first.
    marked = [c for c in dict.fromkeys(cfg.exclude) if c in raw.ch_names]
    if marked:
        if len(marked) >= len(raw.ch_names):
            raise ValueError(
                "excluding those channels would leave nothing to analyse "
                f"({len(marked)} of {len(raw.ch_names)} channels)")
        raw.drop_channels(marked)
        steps.append(f"dropped {len(marked)} channels the reviewer marked bad: "
                     f"{', '.join(sorted(marked))}")

    # 2/3. filtering ------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.highpass or cfg.lowpass:
            design = filter_description(cfg, float(raw.info["sfreq"]))
            if cfg.filter_method == "iir":
                raw.filter(l_freq=cfg.highpass or None, h_freq=cfg.lowpass,
                           method="iir", iir_params=_iir_params(cfg), phase="zero",
                           picks="all", verbose="ERROR")
            else:
                extra = {}
                if cfg.transition_bandwidth:
                    extra = {"l_trans_bandwidth": float(cfg.transition_bandwidth),
                             "h_trans_bandwidth": float(cfg.transition_bandwidth)}
                raw.filter(l_freq=cfg.highpass or None, h_freq=cfg.lowpass,
                           fir_design="firwin", phase=cfg.filter_phase, picks="all",
                           verbose="ERROR", **extra)
            if cfg.highpass:
                steps.append(f"high-pass {cfg.highpass:g} Hz ({design})")
            if cfg.lowpass:
                steps.append(f"low-pass {cfg.lowpass:g} Hz ({design})")
        line_freq = cfg.line_freq if cfg.line_freq is not None else rec.line_freq
        if cfg.notch:
            source = "configured" if cfg.line_freq is not None else "from the dataset"
            nyq = raw.info["sfreq"] / 2.0
            freqs = ([f for f in np.arange(line_freq, nyq, line_freq) if f < 0.9 * nyq]
                     if cfg.notch_harmonics
                     else ([line_freq] if line_freq < 0.9 * nyq else []))
            if freqs:
                raw.notch_filter(freqs=freqs, notch_widths=cfg.notch_width,
                                 fir_design="firwin", phase="zero", picks="all",
                                 verbose="ERROR")
                harmonics = " + harmonics" if cfg.notch_harmonics else " (fundamental only)"
                steps.append(f"notch {line_freq:g} Hz ({source}){harmonics} "
                             f"({', '.join(f'{f:g}' for f in freqs)} Hz, "
                             f"{cfg.notch_width:g} Hz wide)")

    if cfg.resample and abs(float(cfg.resample) - raw.info["sfreq"]) > 1e-9:
        target = float(cfg.resample)
        before = float(raw.info["sfreq"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.resample(target, npad="auto", verbose="ERROR")
        steps.append(f"resampled {before:g} Hz -> {target:g} Hz"
                     + (" (upsampled; adds no information)" if target > before else ""))

    # 3a. regression of a reference or ECG channel ------------------------
    if regress:
        brain = [c for c in raw.ch_names if c not in regress]
        betas = _regress_out(raw, brain, regress)
        raw.drop_channels(regress)
        strength = np.abs(betas)
        steps.append(f"regressed {', '.join(regress)} out of {len(brain)} channels by least "
                     f"squares (largest coefficient {strength.max():.2f}, median "
                     f"{np.median(strength):.2f}); the regressed channel(s) not analysed")

    # 3b. ICA, experimental: fitted and shown; applied only as chosen ------
    ica_record = _fit_ica(raw, cfg, rec, steps) if cfg.ica else None

    # 3c. artifact annotation, on the filtered monopolar signal ------------
    annotations = (_annotate(raw, cfg, float(rec.t_offset), steps)
                   if (cfg.annotate_muscle or cfg.annotate_amplitude) else [])

    data = raw.get_data(picks="all") * 1e6  # volts -> microvolts
    names = list(raw.ch_names)

    # 4. montage ----------------------------------------------------------
    scheme = effective_reference(cfg)
    pairs: list[tuple[str, str]] = []
    montage = "monopolar"
    if scheme == "bipolar":
        pairs = bipolar_pairs(names, exclude=set())
        if not pairs:
            steps.append("bipolar montage requested but no consecutive contact pairs were found; "
                         "kept the original (monopolar) channels")
        else:
            idx = {n: i for i, n in enumerate(names)}
            data = np.stack([data[idx[a]] - data[idx[b]] for a, b in pairs])
            # A pair is marked wherever either of its contacts is.
            renamed = []
            for note in annotations:
                if note["channel"] is None:
                    renamed.append(note)
                    continue
                for a, b in pairs:
                    if note["channel"] in (a, b):
                        renamed.append({**note, "channel": f"{a}-{b}"})
            annotations = renamed
            names = [f"{a}-{b}" for a, b in pairs]
            montage = "bipolar"
            steps.append(f"bipolar montage: {len(pairs)} pairs of neighbouring contacts")
    elif scheme == "average" and len(names) > 1:
        data = data - data.mean(axis=0, keepdims=True)
        montage = "average"
        steps.append(
            f"common average reference across {len(names)} channels — note "
            f"that this shares every channel's noise with every other, which "
            f"is the effect the bipolar montage exists to avoid for HFO work")
    elif scheme == "median" and len(names) > 1:
        data = data - np.median(data, axis=0, keepdims=True)
        montage = "median"
        steps.append(f"common median reference across {len(names)} channels — the "
                     f"average's robust cousin: one faulty contact cannot drag it, "
                     f"but it still shares the common noise")
    elif scheme == "shaft" and len(names) > 1:
        shafts: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            shafts.setdefault(_shaft_of(name), []).append(i)
        referenced = data.copy()
        lonely = 0
        for members in shafts.values():
            if len(members) < 2:
                lonely += 1
                continue
            referenced[members] = data[members] - data[members].mean(axis=0, keepdims=True)
        data = referenced
        montage = "shaft"
        steps.append(f"per-shaft average reference: each contact minus the mean of its "
                     f"own electrode ({len(shafts)} shafts"
                     + (f"; {lonely} single-contact shaft(s) left as recorded" if lonely else "")
                     + ")")

    elif scheme == "laplacian" and len(names) > 1:
        located: dict = {}
        if _has_positions(raw):
            located.update({ch["ch_name"].upper(): np.asarray(ch["loc"][:3], dtype=float)
                            for ch in raw.info["chs"]})
        given = {str(k).upper(): np.asarray(v, dtype=float)
                 for k, v in (positions or {}).items()}
        located.update(given)      # the reader's file over the recording's own
        neighbours, how = laplacian_neighbours(
            names, located or None, cfg.grid_columns,
            source={n: positions_from for n in given} if positions_from else None)
        idx = {n: i for i, n in enumerate(names)}
        referenced = data.copy()
        for name, near in neighbours.items():
            referenced[idx[name]] = data[idx[name]] - data[[idx[n] for n in near]].mean(axis=0)
        data = referenced
        montage = "laplacian"
        alone = len(names) - len(neighbours)
        steps.append("Laplacian reference: each contact minus the mean of its neighbours on "
                     f"its own lead ({len(neighbours)} of {len(names)} contacts"
                     + (f"; {alone} with no neighbour present left as recorded" if alone else "")
                     + "). " + "; ".join(how)
                     + ". A noisy neighbour is shared by every contact beside it.")

    prepared = Prepared(data=np.ascontiguousarray(data, dtype=np.float64), ch_names=names,
                        sfreq=float(raw.info["sfreq"]), t_offset=rec.t_offset, montage=montage,
                        line_freq=float(line_freq),
                        pairs=pairs, steps=steps, recording=rec, annotations=annotations,
                        ica=ica_record)
    if verbose:
        print(f"[onset-hfo] preprocessed: {prepared.n_channels} {montage} channels, "
              f"{prepared.duration:.1f} s @ {prepared.sfreq:g} Hz")
        for s in steps:
            print(f"[onset-hfo]   - {s}")
    return prepared
