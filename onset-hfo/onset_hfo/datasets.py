"""Loading a slice of a public intracranial EEG recording.

Why a *slice*: the example recording is 98 channels x 1000 Hz x ~4.5 minutes
(~105 MB), and other runs in the same dataset are much longer. HFO analysis
does not need the whole file to be demonstrated, and a notebook that downloads
20 MB starts in seconds. The archive is plain HTTPS over S3, so we ask for a
byte range of the binary file and keep the (tiny) text headers intact.

The data
--------
OpenNeuro ``ds003029`` -- "Epilepsy-iEEG-Multicenter-Dataset", CC0 licensed,
BIDS-iEEG, four clinical centres, ECoG and SEEG, with clinician markers for
electrographic seizure onset and offset. See ``docs/DATA.md`` for the full
description and for what the annotations do and do not contain.

The BrainVision format used by the archive stores samples MULTIPLEXED
(ch1t1, ch2t1, ... chNt1, ch1t2, ...) in a headerless binary file, so byte
offset ``k * n_channels * bytes_per_sample`` is exactly sample ``k``. That is
the whole trick behind :func:`fetch_slice`.

Everything returned by this module is wrapped in a :class:`Recording`, which
carries the data *and* the provenance needed to cite it later.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import mne
import pandas as pd

from onset_hfo.config import (
    DATA_CACHE,
    DATASET,
    DATASETS,
    DEFAULT_SUBJECT,
    DEFAULT_TSTART,
    DEFAULT_TSTOP,
    DatasetSpec,
)

__all__ = [
    "Recording",
    "fetch_slice",
    "load_example",
    "list_runs",
    "parse_marked_contacts",
    "expand_contact_ranges",
    "seizure_marker_kind",
    "list_subjects",
    "read_tsv_text",
    "OfflineError",
    "offline",
]

_BYTES_PER_SAMPLE = {"IEEE_FLOAT_32": 4, "INT_16": 2, "UINT_16": 2, "IEEE_FLOAT_64": 8}

# Marker wording is not standardised across the four centres in this archive
# -- the dataset README says so, and a cohort run is where it bites. Observed
# spellings, one per centre:
#
#   NIH  (pt*)    "onset"                  / "offset"
#   UMMC (ummc*)  "sz onset"               / "sz offset"
#   UMF  (umf*)   "eeg sz start"           / "eeg sz end"
#   JHH  (jh*)    "SZ EVENT # (EEG SZ)"    / -- no offset marker at all
#
# Matching only the first two silently drops half the cohort, with no error:
# those subjects simply report "no seizure marked" and every rate-change
# feature comes back empty.
_ONSET_RE = re.compile(
    r"\b(?:(?:sz|seizure)\s*)?(?:eeg\s+|clinical\s+|electrographic\s+)?"
    r"(?:onset|(?:sz\s+|seizure\s+)?start)\b"
    r"|\bsz\s+event\b",
    re.IGNORECASE)
_OFFSET_RE = re.compile(
    r"\b(?:(?:sz|seizure)\s*)?(?:eeg\s+|clinical\s+|electrographic\s+)?"
    r"(?:offset|(?:sz\s+|seizure\s+)?end)\b",
    re.IGNORECASE)
# JHH writes both electrographic and pushbutton (patient-pressed) seizure
# events with the same "SZ EVENT #" prefix. They are different times and mean
# different things -- a pushbutton press follows the EEG change, sometimes by
# many seconds -- so prefer the electrographic one and record which was used.
_ELECTROGRAPHIC_RE = re.compile(r"\b(?:eeg|electrographic)\b", re.IGNORECASE)
_PUSHBUTTON_RE = re.compile(r"\b(?:pb|push\s*button|clinical)\b", re.IGNORECASE)
#: Markers that look like an onset but are recording bookkeeping, not a seizure.
_NOT_A_SEIZURE_RE = re.compile(r"\brec\s+start\b|\bsegment\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# The container every other module works with
# --------------------------------------------------------------------------


@dataclass
class Recording:
    """One continuous stretch of iEEG plus the metadata needed to cite it.

    Attributes
    ----------
    raw:
        An MNE ``Raw`` object, preloaded, in volts (MNE's internal unit).
        ``raw.times[0]`` is 0; see ``t_offset`` for where that sits in the
        original file.
    t_offset:
        Seconds from the start of the original recording to ``raw.times[0]``.
        Every time this package reports is converted back to *original
        recording time* by adding this offset, so that a window quoted in a
        report can be found again in the archive file.
    seizure:
        ``(onset, offset)`` in original recording time, from the clinician
        markers in the dataset, or ``(None, None)`` when the run has none.
    marked_contacts:
        Contacts named by the clinician in free-text markers around seizure
        onset. A weak, textual reference -- NOT a curated seizure-onset-zone
        label. See ``docs/DATA.md``.
    ground_truth:
        Expert or implanted events, when the recording has them: the synthetic
        simulator always, and ``ds003498`` -- whose authors marked HFOs channel
        by channel -- for real data. ``None`` for a recording nobody has
        annotated, such as anything in ``ds003029``.
    reviewed_channels:
        Channels an annotator actually examined. Crucial and easy to miss: in
        ``ds003498`` only a subset of channels was reviewed (the paper kept the
        three most mesial bipolar channels in temporal-lobe cases), so a
        detection on an unreviewed channel is not a false positive -- it is
        unjudged. Evaluation restricts itself to these channels.
    line_freq:
        Mains frequency at the recording site, carried from the dataset so the
        notch filter is right without the caller having to remember.
    """

    raw: mne.io.BaseRaw
    source: str
    subject: str
    task: str
    run: str
    t_offset: float = 0.0
    seizure: tuple[float | None, float | None] = (None, None)
    bads: list[str] = field(default_factory=list)
    events: pd.DataFrame | None = None
    channels: pd.DataFrame | None = None
    marked_contacts: list[str] = field(default_factory=list)
    #: What kind of marker ``seizure`` came from: ``"electrographic"``,
    #: ``"pushbutton"``, ``"unspecified"`` or ``"none"``. A pushbutton press is
    #: when someone reacted, which can trail the EEG change by many seconds, so
    #: a rate change measured against one means something weaker.
    seizure_marker: str = "none"
    ground_truth: pd.DataFrame | None = None
    reviewed_channels: list[str] = field(default_factory=list)
    line_freq: float = 60.0
    dataset_id: str = ""
    citation: str = ""
    notes: list[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    @property
    def sfreq(self) -> float:
        return float(self.raw.info["sfreq"])

    @property
    def duration(self) -> float:
        return float(self.raw.n_times) / self.sfreq

    @property
    def ch_names(self) -> list[str]:
        return list(self.raw.ch_names)

    @property
    def label(self) -> str:
        """Short human-readable id used in reports and figures."""
        return f"{self.subject}/{self.task}/run-{self.run}"

    def abs_time(self, t: float) -> float:
        """Convert a time in this slice to time in the original recording."""
        return float(t) + self.t_offset

    def is_ictal(self, t_abs: float) -> bool:
        """True if an absolute time falls inside the marked seizure."""
        on, off = self.seizure
        return on is not None and off is not None and on <= t_abs <= off

    def provenance(self) -> dict:
        """A JSON-safe description of where this data came from."""
        return {
            "source": self.source,
            "dataset": self.dataset_id,
            "subject": self.subject,
            "task": self.task,
            "run": self.run,
            "sfreq_hz": self.sfreq,
            "n_channels": len(self.ch_names),
            "slice_start_s": self.t_offset,
            "slice_stop_s": self.t_offset + self.duration,
            "seizure_onset_s": self.seizure[0],
            "seizure_offset_s": self.seizure[1],
            "seizure_marker": self.seizure_marker,
            "bad_channels": list(self.bads),
            "line_freq_hz": self.line_freq,
            "n_expert_events": (0 if self.ground_truth is None else int(len(self.ground_truth))),
            "reviewed_channels": list(self.reviewed_channels),
            "citation": self.citation,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# HTTP helpers (stdlib-friendly; ``requests`` is used when available)
# --------------------------------------------------------------------------


class OfflineError(RuntimeError):
    """Raised instead of a network call when offline mode is on."""


#: Set this environment variable to refuse every outbound request from this
#: package. The test suite sets it for the whole session, because "these tests
#: are offline" is a claim that has to be enforced rather than intended: the
#: suite silently downloaded the clinical spreadsheet on every run for weeks,
#: which cost nothing but made CI depend on S3 being up.
OFFLINE_ENV = "ONSET_HFO_OFFLINE"


def offline() -> bool:
    """True when outbound requests are disabled for this process."""
    return os.environ.get(OFFLINE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _http_get(url: str, byte_range: tuple[int, int] | None = None, timeout: float = 120.0,
              retries: int = 4) -> bytes:
    """GET a URL, optionally a byte range. Raises ``RuntimeError`` on failure.

    This is the only place in the package that reaches the network, which is
    what makes :data:`OFFLINE_ENV` a guarantee rather than a convention.

    Transport-level failures are retried with exponential backoff, because
    the alternative is worse than slow: callers that sweep a cohort
    (:mod:`onset_hfo.benchmark`, :mod:`onset_hfo.outcome`,
    :mod:`onset_hfo.stability`) skip a subject that fails and carry on, so a
    single dropped TLS connection silently changes which patients an analysis
    was computed over. One transient SSL EOF is how a 20-patient cohort
    quietly becomes 19 in one window of a stability curve and nowhere else.

    An HTTP status error (a genuinely missing file) is *not* retried -- it
    will not succeed the second time, and pretending otherwise just makes the
    failure slower.
    """
    if offline():
        raise OfflineError(
            f"{OFFLINE_ENV} is set, so this request was refused rather than sent: {url}")
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            import time
            time.sleep(min(2.0 ** attempt, 16.0))
        try:
            return _http_get_once(url, headers, timeout)
        except (OfflineError, RuntimeError):
            raise
        except Exception as exc:  # transport-level: SSL EOF, reset, DNS, timeout
            last = exc
            print(f"[onset-hfo] retrying {url.rsplit('/', 1)[-1]} after "
                  f"{type(exc).__name__} (attempt {attempt + 1}/{retries + 1})")
    raise RuntimeError(f"{url} failed after {retries + 1} attempts: {last}") from last


def _http_get_once(url: str, headers: dict, timeout: float) -> bytes:
    try:  # requests gives nicer proxy handling when it is installed
        import requests

        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")
        return resp.content
    except ImportError:  # pragma: no cover - exercised only without requests
        import urllib.request

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return fh.read()


def _spec(dataset: str | DatasetSpec | None) -> DatasetSpec:
    """Resolve a dataset id (or a spec) to its :class:`DatasetSpec`."""
    if isinstance(dataset, DatasetSpec):
        return dataset
    if dataset is None:
        return DATASET
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset {dataset!r}; known: {', '.join(DATASETS)}") from None


def _dataset_url(spec: DatasetSpec, *parts: str) -> str:
    return "/".join([spec.base_url, spec.dataset_id, *parts])


def _stem(spec: DatasetSpec, subject: str, run: str, session: str | None = None,
          task: str | None = None, acq: str | None = None) -> str:
    """Build the BIDS path for one run, using only the entities that exist.

    Archives differ: ``ds003029`` names files
    ``sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01``, while ``ds003498``
    names them ``sub-01_ses-interictalsleep_run-01`` with no task or acq at
    all. Hard-coding either shape is how a loader ends up serving one dataset.
    """
    session = session if session is not None else spec.session
    task = task if task is not None else spec.task
    acq = acq if acq is not None else spec.acq
    parts = [subject]
    if session:
        parts.append(session)
    if task:
        parts.append(f"task-{task}")
    if acq:
        parts.append(f"acq-{acq}")
    parts.append(f"run-{run}")
    folder = f"{subject}/{session}/ieeg" if session else f"{subject}/ieeg"
    return f"{folder}/{'_'.join(parts)}"


# --------------------------------------------------------------------------
# BrainVision header parsing
# --------------------------------------------------------------------------


def _parse_vhdr(text: str) -> dict:
    """Extract the few header fields we need to compute byte offsets."""
    info: dict = {"channels": []}
    section = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if section == "channel infos" and key.lower().startswith("ch"):
            parts = value.split(",")
            info["channels"].append(parts[0].replace(r"\1", ","))
        else:
            info[key.strip()] = value.strip()
    if info.get("DataOrientation", "MULTIPLEXED").upper() != "MULTIPLEXED":
        raise RuntimeError("Only MULTIPLEXED BrainVision files can be sliced by byte range")
    info["n_channels"] = int(info["NumberOfChannels"])
    info["sfreq"] = 1e6 / float(info["SamplingInterval"])
    fmt = info.get("BinaryFormat", "IEEE_FLOAT_32").upper()
    if fmt not in _BYTES_PER_SAMPLE:
        raise RuntimeError(f"Unsupported BrainVision binary format: {fmt}")
    info["bytes_per_sample"] = _BYTES_PER_SAMPLE[fmt]
    return info


_MINIMAL_VMRK = """Brain Vision Data Exchange Marker File, Version 1.0

[Common Infos]
Codepage=UTF-8
DataFile={data_file}

[Marker Infos]
Mk1=New Segment,,1,1,0,00000000000000000000
"""


# --------------------------------------------------------------------------
# The public entry points
# --------------------------------------------------------------------------


def list_runs(subject: str = DEFAULT_SUBJECT, session: str | None = None,
              dataset: str | None = None) -> pd.DataFrame:
    """List the iEEG runs available for a subject in the public archive.

    Returns a table with ``task``, ``acq``, ``run`` and ``size_mb``; the task
    and acq columns are empty for archives whose filenames omit them.
    """
    spec = _spec(dataset)
    session = session if session is not None else spec.session
    prefix = f"{spec.dataset_id}/{subject}/{session}/ieeg/" if session \
        else f"{spec.dataset_id}/{subject}/ieeg/"
    url = f"{spec.base_url}/?list-type=2&prefix={prefix}&max-keys=1000"
    xml = _http_get(url).decode("utf-8", "replace")
    rows = []
    for key, size in re.findall(r"<Key>([^<]+)</Key>\s*<LastModified>[^<]*</LastModified>\s*"
                                r"<ETag>[^<]*</ETag>\s*<Size>(\d+)</Size>", xml):
        if not key.endswith("_ieeg.eeg"):
            continue
        name = Path(key).name
        run = re.search(r"run-(\d+)", name)
        if not run:
            continue
        task = re.search(r"task-([a-zA-Z0-9]+)", name)
        acq = re.search(r"acq-([a-zA-Z0-9]+)", name)
        rows.append({"subject": subject, "task": task.group(1) if task else "",
                     "acq": acq.group(1) if acq else "", "run": run.group(1),
                     "size_mb": round(int(size) / 1e6, 1)})
    table = pd.DataFrame(rows)
    return table.sort_values(["task", "run"]).reset_index(drop=True) if len(table) else table


def fetch_slice(
    subject: str = DEFAULT_SUBJECT,
    task: str | None = None,
    run: str | None = None,
    t_start: float = DEFAULT_TSTART,
    t_stop: float = DEFAULT_TSTOP,
    session: str | None = None,
    acq: str | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
    verbose: bool = True,
    dataset: str | None = None,
) -> Recording:
    """Download (once) and load ``[t_start, t_stop)`` of a public iEEG run.

    Parameters
    ----------
    subject, task, run, session, acq:
        BIDS entities identifying the recording; the defaults point at the
        example used everywhere in the documentation.
    t_start, t_stop:
        Seconds from the start of the original file. Only this range is
        downloaded. Keep it short: 60 s of a 98-channel 1000 Hz float32
        recording is ~24 MB.
    cache_dir:
        Where the slice is stored. Defaults to ``artifacts/data``.
    force:
        Re-download even if the slice is already cached.

    Returns
    -------
    Recording
        With ``t_offset = t_start``, so reported times are original-file times.

    Raises
    ------
    RuntimeError
        If the archive cannot be reached. Use :func:`onset_hfo.synthetic.make_synthetic_recording`
        for an offline demonstration, or see ``docs/DATA.md`` for how to point
        the loader at a local BIDS file instead.
    """
    if t_stop <= t_start:
        raise ValueError("t_stop must be greater than t_start")
    spec = _spec(dataset)
    session = session if session is not None else spec.session
    task = task if task is not None else spec.task
    acq = acq if acq is not None else spec.acq
    run = run if run is not None else spec.default_run
    cache_dir = Path(cache_dir or DATA_CACHE) / spec.dataset_id
    tag = f"task-{task}_" if task else ""
    slice_dir = cache_dir / f"{subject}_{tag}run-{run}_{t_start:g}-{t_stop:g}s"
    meta_path = slice_dir / "slice.json"
    stem = _stem(spec, subject, run, session, task, acq)

    if force or not meta_path.exists():
        slice_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[onset-hfo] fetching {subject} {task or spec.dataset_id} run-{run} "
                  f"[{t_start:g}, {t_stop:g}) s from OpenNeuro {spec.dataset_id}")
        vhdr_text = _http_get(_dataset_url(spec, f"{stem}_ieeg.vhdr")).decode("utf-8", "replace")
        header = _parse_vhdr(vhdr_text)
        sfreq = header["sfreq"]
        frame = header["n_channels"] * header["bytes_per_sample"]
        first = int(round(t_start * sfreq)) * frame
        last = int(round(t_stop * sfreq)) * frame - 1
        mb = (last - first + 1) / 1e6
        if mb > 250 and verbose:
            warnings.warn(f"Requested slice is {mb:.0f} MB; consider a shorter window.", stacklevel=2)
        if verbose:
            print(f"[onset-hfo]   {header['n_channels']} channels @ {sfreq:g} Hz -> {mb:.1f} MB")
        payload = _http_get(_dataset_url(spec, f"{stem}_ieeg.eeg"), byte_range=(first, last))
        # Guard against a proxy that ignores Range and returns the whole file.
        expected = last - first + 1
        if len(payload) > expected:
            payload = payload[first:last + 1] if len(payload) > last else payload[:expected]
        usable = (len(payload) // frame) * frame
        data_name = header.get("DataFile", f"{Path(stem).name}_ieeg.eeg")
        (slice_dir / data_name).write_bytes(payload[:usable])

        marker_name = header.get("MarkerFile", data_name.replace(".eeg", ".vmrk"))
        (slice_dir / marker_name).write_text(_MINIMAL_VMRK.format(data_file=data_name), "utf-8")
        (slice_dir / f"{Path(stem).name}_ieeg.vhdr").write_text(vhdr_text, "utf-8")

        for side in ("channels.tsv", "events.tsv"):
            try:
                blob = _http_get(_dataset_url(spec, f"{stem}_{side}"))
                (slice_dir / side).write_bytes(blob)
            except Exception:  # the archive does not always ship events.tsv
                pass
        meta_path.write_text(json.dumps({
            "dataset": spec.dataset_id, "subject": subject, "session": session, "task": task,
            "acq": acq, "run": run, "t_start": t_start, "t_stop": t_stop,
            "sfreq": sfreq, "n_channels": header["n_channels"],
            "vhdr": f"{Path(stem).name}_ieeg.vhdr",
        }, indent=2), "utf-8")
    elif verbose:
        print(f"[onset-hfo] using cached slice {slice_dir}")

    meta = json.loads(meta_path.read_text())
    return _load_cached_slice(slice_dir, meta, verbose=verbose)


def _load_cached_slice(slice_dir: Path, meta: dict, verbose: bool = True) -> Recording:
    """Read a cached slice into a :class:`Recording` (no network access)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_brainvision(slice_dir / meta["vhdr"], preload=True, verbose="ERROR")

    channels = _read_tsv(slice_dir / "channels.tsv")
    events = _read_tsv(slice_dir / "events.tsv")
    t_start = float(meta["t_start"])

    bads: list[str] = []
    if channels is not None and {"name", "status"} <= set(channels.columns):
        bads = [str(n) for n, s in zip(channels["name"], channels["status"], strict=False) if str(s).lower() == "bad"]
        bads = [b for b in bads if b in raw.ch_names]
        raw.info["bads"] = list(bads)
        types = {str(n): _mne_type(str(t)) for n, t in zip(channels["name"], channels.get("type", []), strict=False)}
        types = {k: v for k, v in types.items() if k in raw.ch_names}
        if types:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_channel_types(types, verbose="ERROR")

    spec = _spec(meta.get("dataset"))
    seizure_onset, seizure_offset, marker_kind = _seizure_times(events, with_kind=True)
    seizure = (seizure_onset, seizure_offset)
    marked = parse_marked_contacts(events, raw.ch_names, seizure)
    _attach_annotations(raw, events, t_start)

    notes = [
        f"slice {meta['t_start']:g}-{meta['t_stop']:g} s of the original recording; "
        "all times reported by this pipeline are in original-recording seconds",
        f"{len(bads)} channels flagged bad in channels.tsv "
        "(white matter, CSF, outside brain, or noisy) and excluded",
    ]
    if raw.info["sfreq"] < 600:
        notes.append(f"sampling rate {raw.info['sfreq']:g} Hz: ripples (80-250 Hz) only, "
                     "fast ripples (250-500 Hz) are not analysable in this recording")
    notes.append(f"mains frequency {spec.line_freq:g} Hz ({spec.dataset_id})")

    truth = reviewed = None
    if spec.has_hfo_annotations:
        truth, reviewed = parse_hfo_annotations(events, window=(t_start, float(meta["t_stop"])))
        notes.append(
            f"expert HFO markings: {len(truth)} events on {len(reviewed)} reviewed channels "
            f"({', '.join(reviewed[:6])}{', ...' if len(reviewed) > 6 else ''})")
        notes.append("only those channels were annotated, so a detection elsewhere is unjudged "
                     "rather than wrong; evaluation is restricted to them")

    rec = Recording(
        raw=raw,
        source=f"openneuro:{meta['dataset']}",
        subject=meta["subject"],
        task=meta["task"],
        run=meta["run"],
        t_offset=t_start,
        seizure=seizure,
        bads=bads,
        events=events,
        channels=channels,
        marked_contacts=marked,
        seizure_marker=marker_kind,
        ground_truth=truth,
        reviewed_channels=reviewed or [],
        line_freq=spec.line_freq,
        dataset_id=spec.dataset_id,
        citation=spec.citation,
        notes=notes,
    )
    if verbose:
        on, off = seizure
        span = f"{on:.1f}-{off:.1f} s" if on is not None and off is not None else "none marked"
        print(f"[onset-hfo] loaded {rec.label}: {len(rec.ch_names)} channels, "
              f"{rec.duration:.1f} s @ {rec.sfreq:g} Hz; seizure {span}")
    return rec


def iter_slices(t_start: float, t_stop: float, chunk_s: float,
                overlap_s: float = 2.0, verbose: bool = True, **kwargs):
    """Yield ``(ChunkPlan, Recording)`` over a long window, one chunk at a time.

    The chunk source for :mod:`onset_hfo.streaming`. Each chunk is an ordinary
    :func:`fetch_slice` call, so the byte-range fetching, the caching and the
    BrainVision parsing are the same code the single-slice path uses -- there
    is no second loader to keep in step.

    Memory stays flat: one chunk is loaded, yielded and dropped before the
    next is fetched. **Disk does not.** Every chunk is cached under
    ``artifacts/data`` exactly as a single slice would be, so an hour of
    recording leaves an hour of recording on disk. That is a deliberate
    trade -- re-running an analysis should not re-download it -- and it is
    stated here because "streaming" usually implies otherwise.
    """
    from onset_hfo.streaming import plan_chunks

    for plan in plan_chunks(t_start, t_stop, chunk_s, overlap_s):
        yield plan, fetch_slice(t_start=plan.read_start, t_stop=plan.read_stop,
                                verbose=verbose, **kwargs)


def load_example(**kwargs) -> Recording:
    """The one-liner used by the notebooks: the default example slice."""
    return fetch_slice(**kwargs)


# --------------------------------------------------------------------------
# Sidecar parsing
# --------------------------------------------------------------------------


def list_subjects(dataset: str | None = None) -> list[str]:
    """Every ``sub-*`` directory in the archive that contains a signal file.

    One S3 listing, no signal downloaded. Used to plan a cohort run before
    committing to the bandwidth. Reading the bucket rather than
    ``participants.tsv`` on purpose: the two disagree when a participant has a
    row but no recordings, and it is the recordings that can be analysed.
    """
    spec = _spec(dataset)
    subjects: set[str] = set()
    token = ""
    for _page in range(20):
        url = (f"{spec.base_url}/?list-type=2&prefix={spec.dataset_id}/"
               f"&max-keys=1000")
        if token:
            from urllib.parse import quote
            url += f"&continuation-token={quote(token, safe='')}"
        xml = _http_get(url).decode("utf-8", "replace")
        for key in re.findall(r"<Key>([^<]+)</Key>", xml):
            if key.endswith("_ieeg.eeg"):
                match = re.search(r"/(sub-[A-Za-z0-9]+)/", key)
                if match:
                    subjects.add(match.group(1))
        if "<IsTruncated>true</IsTruncated>" not in xml:
            break
        found = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not found:
            break
        token = found.group(1)
    return sorted(subjects)


def read_tsv_text(text: str) -> pd.DataFrame | None:
    """Parse a TSV already in memory (a sidecar fetched without caching it)."""
    import io

    try:
        frame = pd.read_csv(io.StringIO(text), sep="\t")
    except Exception:
        return None
    return frame if len(frame.columns) else None


def _read_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t", dtype=str).rename(columns=str.strip)
    except Exception:
        return None


def _mne_type(bids_type: str) -> str:
    return {"ECOG": "ecog", "SEEG": "seeg", "EEG": "eeg", "EKG": "ecg", "ECG": "ecg",
            "EMG": "emg", "EOG": "eog", "MISC": "misc", "TRIG": "stim"}.get(
        str(bids_type).upper(), "misc")


def seizure_marker_kind(label: str) -> str:
    """``"electrographic"``, ``"pushbutton"`` or ``"unspecified"`` for one marker.

    Kept public because the distinction changes what a rate-change number
    means: a pushbutton press is when the patient or nurse reacted, which can
    trail the EEG change by many seconds.
    """
    if _ELECTROGRAPHIC_RE.search(label):
        return "electrographic"
    if _PUSHBUTTON_RE.search(label):
        return "pushbutton"
    return "unspecified"


def _seizure_times(events: pd.DataFrame | None,
                   with_kind: bool = False):
    """Find seizure onset/offset in the BIDS events table.

    Marker wording is not standardised across centres (the dataset README says
    so explicitly), hence the regular expressions rather than an exact match --
    see the patterns above for the four spellings this archive actually uses.

    When a run carries several onset markers, the **electrographic** one wins
    over a pushbutton one regardless of which came first in the file. With
    ``with_kind=True`` the marker kind is returned alongside the times, so a
    caller can record in its provenance that a given rate change was measured
    against a patient-pressed button rather than an EEG change.
    """
    empty = (None, None, "none") if with_kind else (None, None)
    if events is None or "trial_type" not in events.columns:
        return empty
    candidates: list[tuple[float, str]] = []
    offset = None
    for _, row in events.iterrows():
        try:
            t = float(row["onset"])
        except (TypeError, ValueError):
            continue
        label = str(row["trial_type"])
        if _NOT_A_SEIZURE_RE.search(label):
            continue
        if _ONSET_RE.search(label):
            candidates.append((t, seizure_marker_kind(label)))
        elif _OFFSET_RE.search(label):
            offset = t
    if not candidates:
        return empty
    # Electrographic first, then the earliest of whatever is left.
    electrographic = [c for c in candidates if c[1] == "electrographic"]
    onset, kind = min(electrographic or candidates, key=lambda c: c[0])
    if offset is not None and offset <= onset:
        offset = None
    return (onset, offset, kind) if with_kind else (onset, offset)


#: ``ripple_HL2-3`` -> kind "ripple", contacts HL2 and HL3.
_HFO_LABEL = re.compile(r"^(ripple|fr|frandr)_([A-Za-z]+[A-Za-z\']*?)(\d{1,3})-(\d{1,3})$")
#: Some labels name a single contact rather than a bipolar pair.
_HFO_LABEL_SINGLE = re.compile(r"^(ripple|fr|frandr)_([A-Za-z]+[A-Za-z\']*?\d{1,3})$")

#: What each label means, as a band this pipeline can actually detect in.
_LABEL_BANDS = {"ripple": ("ripple",), "fr": ("fast_ripple",),
                "frandr": ("ripple", "fast_ripple")}


def parse_hfo_annotations(events: pd.DataFrame | None,
                          window: tuple[float, float] | None = None
                          ) -> tuple[pd.DataFrame, list[str]]:
    """Expert HFO markings from a BIDS ``events.tsv`` into a truth table.

    The labels look like ``ripple_HL2-3``, ``fr_PHR1-2`` or ``frandr_AHR3-4``:
    the kind of oscillation, then the channel, which is a *bipolar pair*
    written with the shared electrode name once. We expand that to this
    package's channel naming (``HL2-HL3``) so detections and markings can be
    compared without a mapping table.

    A ``frandr`` event -- a fast ripple riding on a ripple -- becomes **two
    rows**, one per band. That is not double counting: the event really does
    carry evidence in both bands, and each band is scored separately.

    Returns
    -------
    (truth, reviewed_channels)
        ``truth`` has one row per (event, band) with ``kind``, ``channel``,
        ``contact``, ``start`` and ``stop`` in original-recording seconds.
        ``reviewed_channels`` are the channels that carry at least one
        marking -- the only channels on which a detection can be judged.
    """
    columns = ["kind", "channel", "label", "contact", "contact_b", "start", "stop", "duration_ms"]
    if events is None or "trial_type" not in events.columns:
        return pd.DataFrame(columns=columns), []

    rows: list[dict] = []
    reviewed: list[str] = []
    for _, row in events.iterrows():
        label = str(row.get("trial_type", ""))
        match = _HFO_LABEL.match(label)
        if match:
            kind, lead, first, second = match.groups()
            a, b = f"{lead}{first}", f"{lead}{second}"
            channel = f"{a}-{b}"
        else:
            single = _HFO_LABEL_SINGLE.match(label)
            if not single:
                continue
            kind, a = single.groups()
            b, channel = "", a
        try:
            start = float(row["onset"])
            duration = float(row.get("duration", 0) or 0.0)
        except (TypeError, ValueError):
            continue
        if channel not in reviewed:
            reviewed.append(channel)
        if window is not None and not (window[0] <= start < window[1]):
            continue
        for band in _LABEL_BANDS[kind]:
            rows.append({"kind": band, "channel": channel, "label": label, "contact": a,
                         "contact_b": b, "start": start, "stop": start + duration,
                         "duration_ms": duration * 1000.0})
    truth = pd.DataFrame(rows, columns=columns)
    if len(truth):
        truth = truth.sort_values(["start", "channel"]).reset_index(drop=True)
    return truth, sorted(reviewed)


def parse_marked_contacts(
    events: pd.DataFrame | None,
    ch_names: Iterable[str],
    seizure: tuple[float | None, float | None] = (None, None),
    window_s: float = 30.0,
) -> list[str]:
    """Contacts named in clinician free-text markers near seizure onset.

    The markers look like ``"AD1-4, ATT1,2"`` or ``"G16"``: the reviewer typed
    which contacts were involved. We expand ranges (``AD1-4`` -> AD1..AD4) and
    keep only tokens that match a real channel name.

    This is a *weak textual reference*, useful for a sanity check ("do the
    channels our detector ranks highest appear in what the clinician wrote?"),
    and it is not a curated seizure-onset-zone annotation. Treat any agreement
    as encouraging and any disagreement as uninformative.
    """
    names = {str(c).upper(): str(c) for c in ch_names}
    if events is None or "trial_type" not in events.columns:
        return []
    onset = seizure[0]
    found: list[str] = []
    for _, row in events.iterrows():
        try:
            t = float(row["onset"])
        except (TypeError, ValueError):
            continue
        if onset is not None and not (onset - window_s <= t <= onset + window_s):
            continue
        for token in _expand_contact_tokens(str(row["trial_type"])):
            real = names.get(token.upper())
            if real and real not in found:
                found.append(real)
    return found


def expand_contact_ranges(text: str) -> tuple[list[str], list[str]]:
    """``"AD1-4, ATT1,2"`` -> ``(["AD1".."AD4", "ATT1", "ATT2"], [])``.

    Clinicians write contact lists as shorthand ranges, in free text, in
    several places in these archives: seizure markers in ``events.tsv`` and
    the resected-zone column of the Zurich clinical sheet both use this
    grammar. One parser serves both.

    Returns the expanded contacts **and the chunks that could not be parsed**.
    The second list is the point: real sheets contain typos (``ds003498``
    writes ``1ll22-24`` where it means ``tll22-24``), and a parser that
    silently returns fewer contacts turns a typo into a quietly wrong
    denominator. Callers are expected to surface what came back unparsed.
    """
    out: list[str] = []
    unparsed: list[str] = []
    prefix = None
    for chunk in re.split(r"[,\s]+", text.strip()):
        if not chunk:
            continue
        m = re.fullmatch(r"([A-Za-z]{1,5})(\d{1,3})(?:-(\d{1,3}))?", chunk)
        if m:
            prefix = m.group(1)
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            if hi >= lo and hi - lo < 32:
                out.extend(f"{prefix}{i}" for i in range(lo, hi + 1))
            else:
                unparsed.append(chunk)
            continue
        m = re.fullmatch(r"(\d{1,3})(?:-(\d{1,3}))?", chunk)
        if m and prefix:  # a bare number continues the previous prefix ("ATT1,2")
            lo = int(m.group(1))
            hi = int(m.group(2)) if m.group(2) else lo
            if hi >= lo and hi - lo < 32:
                out.extend(f"{prefix}{i}" for i in range(lo, hi + 1))
            else:
                unparsed.append(chunk)
            continue
        unparsed.append(chunk)
    return out, unparsed


def _expand_contact_tokens(text: str) -> list[str]:
    """The contacts named in a free-text marker, ignoring unparsable chunks."""
    return expand_contact_ranges(text)[0]


def _attach_annotations(raw: mne.io.BaseRaw, events: pd.DataFrame | None, t_start: float) -> None:
    """Put the clinician markers that fall inside the slice onto the Raw object."""
    if events is None or "onset" not in events.columns:
        return
    onsets, durations, descs = [], [], []
    for _, row in events.iterrows():
        try:
            t = float(row["onset"]) - t_start
        except (TypeError, ValueError):
            continue
        if not (0 <= t < raw.n_times / raw.info["sfreq"]):
            continue
        onsets.append(t)
        try:
            durations.append(float(row.get("duration", 0) or 0))
        except (TypeError, ValueError):
            durations.append(0.0)
        descs.append(str(row.get("trial_type", "event")))
    if onsets:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_annotations(mne.Annotations(onsets, durations, descs), verbose="ERROR")


def load_local_brainvision(vhdr: str | Path, **kwargs) -> Recording:
    """Escape hatch: load a BrainVision file you already have on disk.

    Handy when a lab wants to try the pipeline on its own de-identified export
    without touching the public archive. ``kwargs`` are passed to
    :class:`Recording` so you can supply ``subject``, ``seizure`` and so on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    kwargs.setdefault("source", f"local:{Path(vhdr).name}")
    kwargs.setdefault("subject", Path(vhdr).stem)
    kwargs.setdefault("task", "unknown")
    kwargs.setdefault("run", "01")
    return Recording(raw=raw, **kwargs)
