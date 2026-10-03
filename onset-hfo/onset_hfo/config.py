"""Central configuration for the Onset-HFO prototype.

Everything that a reviewer might want to change -- frequency bands, detector
thresholds, the public dataset that ships as the default example, where files
are cached -- lives here, so that no magic number is buried inside an algorithm.

Read this file first: the rest of the package is easier to follow once you know
what ``Bands``, ``DetectorConfig`` and ``DATASET`` mean.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Where things are stored
# --------------------------------------------------------------------------

#: Root of this project (the directory that contains ``onset_hfo/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Downloaded data and pipeline outputs. Override with ONSET_HFO_HOME so that
#: Colab (``/content``) and a laptop can use different locations.
HOME = Path(os.environ.get("ONSET_HFO_HOME", PROJECT_ROOT / "artifacts")).expanduser()

#: Raw data slices downloaded from the public archive are cached here.
DATA_CACHE = HOME / "data"

#: Detection tables, reports and figures are written here.
RESULTS_DIR = HOME / "results"


def ensure_dirs() -> None:
    """Create the cache/results directories. Safe to call repeatedly."""
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# The public dataset used by default
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Identity, access details and quirks of one public dataset.

    Two datasets are wired in. They answer different questions, and the
    difference matters more than any parameter in this file:

    ``ds003029`` is *ictal* -- recordings around seizures, with clinician
    onset markers but no HFO annotations. You can measure rates and detector
    agreement on it; you cannot measure precision or recall.

    ``ds003498`` is *interictal slow-wave sleep* -- the setting the clinical
    HFO literature actually uses -- and it ships expert-validated HFO events
    per channel. That is what makes a real precision/recall number possible.

    Attributes that exist because datasets differ in ways that break analyses
    silently: ``line_freq`` (50 Hz in Zurich, 60 Hz in the US -- notching the
    wrong one leaves the interference in and carves a hole where there was
    none) and the BIDS entities each archive uses in its filenames.
    """

    dataset_id: str
    name: str
    doi: str
    citation: str
    #: Mains frequency at the recording site. Wrong value = wrong notch.
    line_freq: float = 60.0
    #: BIDS entities this dataset's filenames carry. ``None`` means absent.
    session: str | None = None
    task: str | None = None
    acq: str | None = None
    default_run: str = "01"
    #: True when ``*_events.tsv`` carries expert HFO markings.
    has_hfo_annotations: bool = False
    #: Path inside the archive to a clinical sheet naming the resected
    #: contacts per subject, if the archive ships one. ``None`` means the
    #: resected zone is unknown and no outcome study is possible.
    clinical_sheet: str | None = None
    #: True when ``participants.tsv`` carries post-surgical seizure outcome.
    has_outcomes: bool = False
    license: str = "CC0"
    #: Public S3 mirror of the OpenNeuro bucket (no credentials, Range requests).
    base_url: str = "https://s3.amazonaws.com/openneuro.org"


DATASETS: dict[str, DatasetSpec] = {
    "ds003029": DatasetSpec(
        dataset_id="ds003029",
        name="Epilepsy-iEEG-Multicenter-Dataset (Fragility multicenter study)",
        doi="10.18112/openneuro.ds003029.v1.0.3",
        citation=(
            "Li A, Inati S, Zaghloul K, Crone N, Anderson W, Johnson E, Cajigas I, Brusko D, "
            "Jagid J, Claudio A, Kanner A, Hopp J, Chen S, Haagensen J, Sarma S. "
            "Epilepsy-iEEG-Multicenter-Dataset. OpenNeuro (2021). "
            "doi:10.18112/openneuro.ds003029.v1.0.3 -- the archive asks that work using it "
            "also cite 'Neural fragility as an EEG marker of the seizure onset zone', "
            "doi:10.1101/862797 (Nature Neuroscience, 2023)."
        ),
        line_freq=60.0,
        session="ses-presurgery",
        task="ictal",
        acq="ecog",
        default_run="01",
        has_hfo_annotations=False,
    ),
    "ds003498": DatasetSpec(
        dataset_id="ds003498",
        name="Zurich iEEG HFO dataset: interictal slow-wave sleep with expert HFO markings",
        doi="10.18112/openneuro.ds003498.v1.0.1",
        citation=(
            "Fedele T, Burnos S, Boran E, Krayenbuehl N, Hilfiker P, Grunwald T, Sarnthein J. "
            "Resection of high frequency oscillations predicts seizure outcome in the individual "
            "patient. Scientific Reports 7:13836 (2017). doi:10.1038/s41598-017-13064-1 -- "
            "BIDS conversion by A. Zhang (mne-hfo), OpenNeuro ds003498, CC0."
        ),
        line_freq=50.0,          # Zurich
        session="ses-interictalsleep",
        task=None,               # these filenames carry no task or acq entity
        acq=None,
        default_run="01",
        has_hfo_annotations=True,
        clinical_sheet="sourcedata/clinical_ch_sheet_zurich.xlsx",
        has_outcomes=True,
    ),
}

#: The dataset used when none is named: ictal, and the one the quickstart uses.
DATASET = DATASETS["ds003029"]

#: The example recording the quickstart uses. ECoG grid + strips, 1000 Hz,
#: one seizure with clinician onset/offset markers.
DEFAULT_SUBJECT = "sub-pt01"
DEFAULT_SESSION = "ses-presurgery"
DEFAULT_TASK = "ictal"
DEFAULT_ACQ = "ecog"
DEFAULT_RUN = "01"

#: Default slice of that recording, in seconds from the start of the file.
#: 50-110 s spans ~26 s of pre-ictal baseline and ~34 s from the marked
#: electrographic onset (75.95 s), which is enough to show a rate change.
DEFAULT_TSTART = 50.0
DEFAULT_TSTOP = 110.0


# --------------------------------------------------------------------------
# Frequency bands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Bands:
    """Frequency bands used throughout the package (Hz).

    ``ripple`` and ``fast_ripple`` follow the conventional HFO definitions
    (Zijlmans et al., Ann Neurol 2012). ``ied`` is the band in which
    interictal epileptiform discharges (spikes/sharp waves) are detected.
    """

    ripple: tuple[float, float] = (80.0, 250.0)
    fast_ripple: tuple[float, float] = (250.0, 500.0)
    ied: tuple[float, float] = (5.0, 60.0)
    #: Low band used for the artifact check that rejects filter "ringing".
    low: tuple[float, float] = (1.0, 30.0)

    def usable(self, sfreq: float, band: tuple[float, float]) -> bool:
        """True if ``band`` is below the Nyquist frequency with a safety margin.

        A 1000 Hz recording (Nyquist 500 Hz) can carry ripples but cannot
        support honest fast-ripple analysis, so the pipeline skips that band
        instead of producing numbers nobody should trust.
        """
        return band[1] <= 0.9 * (sfreq / 2.0)


BANDS = Bands()


# --------------------------------------------------------------------------
# Detector parameters
# --------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """Parameters shared by every HFO detector.

    Defaults follow the classic references and are intentionally conservative:
    a prototype that reports a few well-formed events is more useful than one
    that reports thousands of filter artifacts.

    Attributes
    ----------
    band:
        Band-pass applied before energy is measured, in Hz.
    rms_window_ms:
        Length of the sliding window used for RMS / line-length, in ms
        (Staba et al. 2002 used 3 ms).
    threshold_sd:
        Detection threshold in robust standard deviations above the channel's
        own baseline (Staba used 5).
    extend_sd:
        Hysteresis. Once the feature crosses ``threshold_sd``, the event is
        extended outwards while the feature stays above ``extend_sd``. Without
        this, only the loudest middle of an oscillation is measured and every
        duration (and every cycle count) is an underestimate.
    peak_threshold_sd:
        Secondary threshold used by the oscillation-count criterion, in robust
        SDs of the band-passed signal. Staba used 3 SDs of a hand-picked
        *baseline* segment; this implementation estimates the SD from the whole
        channel, events included, which is a larger number, so a smaller
        multiplier expresses the same idea. See ``docs/METHODS.md``.
    min_peaks:
        Minimum number of rectified peaks above ``peak_threshold_sd`` inside a
        candidate event (Staba used 6). This is what makes an event an
        *oscillation* rather than a single transient.
    min_duration_ms / max_duration_ms:
        Accepted event duration. Ripples are typically 20-100 ms; the upper
        bound removes sustained artifacts.
    merge_gap_ms:
        Candidates separated by less than this are merged into one event.
    baseline:
        ``"robust"`` uses median + k*1.4826*MAD (recommended: the events
        themselves barely move the estimate); ``"sd"`` uses mean + k*SD, the
        original formulation.
    """

    band: tuple[float, float] = BANDS.ripple
    rms_window_ms: float = 3.0
    threshold_sd: float = 5.0
    extend_sd: float = 2.0
    peak_threshold_sd: float = 2.0
    min_peaks: int = 6
    min_duration_ms: float = 6.0
    max_duration_ms: float = 200.0
    merge_gap_ms: float = 10.0
    baseline: str = "robust"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpikeConfig:
    """Parameters of the interictal epileptiform discharge (IED) detector."""

    band: tuple[float, float] = BANDS.ied
    #: Amplitude threshold on the band-passed signal, in robust SDs.
    threshold_sd: float = 6.0
    #: Sharpness (first-derivative) threshold, in robust SDs of the derivative.
    slope_sd: float = 5.0
    #: Duration is measured at half of the peak amplitude, so these bounds are
    #: narrower than the 20-70 ms a clinician measures at the wave's base.
    min_duration_ms: float = 10.0
    max_duration_ms: float = 200.0
    #: Minimum distance between two accepted discharges.
    refractory_ms: float = 200.0
    #: Reject a candidate when the *unfiltered* signal jumps by more than this
    #: many robust SDs of that channel's sample-to-sample difference. A
    #: discharge is sharp on the millisecond scale; an electrode pop is
    #: discontinuous. Measured on synthetic data, genuine discharges score
    #: 3-5 on this feature and artifact steps score 25-64, so the criterion
    #: costs no recall and lifts precision from 0.58 to ~0.95
    #: (docs/EVALUATION.md).
    max_raw_jump_sd: float = 10.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationConfig:
    """Parameters of the post-detection artifact rejection stage.

    HFO detectors are famous for turning sharp transients into "ripples"
    through filter ringing. These checks look at the *unfiltered* signal in the
    event window and ask whether a genuine narrow-band oscillation is present.
    """

    #: Minimum prominence (dB) of the in-band spectral peak over the local
    #: 1/f background for the event to be accepted. 4 dB was chosen on
    #: synthetic data, where it lifts precision from 0.63 to 0.97 at a cost of
    #: 0.01 recall (docs/EVALUATION.md has the full sweep). A clean implanted
    #: ripple scores above 20 dB, so this is a floor, not a tight filter.
    min_peak_prominence_db: float = 4.0
    #: Reject if the raw low-band (1-30 Hz) amplitude in the window exceeds
    #: this many robust SDs -- the signature of a spike-induced false ripple,
    #: unless a clear spectral peak survives the check above.
    max_low_band_sd: float = 8.0
    #: Reject events with fewer than this many cycles (measured duration x
    #: peak frequency). Four oscillations is the textbook definition of an
    #: HFO, but the measured duration is only the time the envelope stays
    #: above the hysteresis threshold, so the same event counts fewer cycles
    #: here than a reviewer would count by eye. Two is therefore a floor that
    #: removes single transients without discarding short real ripples; the
    #: oscillation-count criterion (``min_peaks``) does the finer work.
    #: Measured on synthetic data: raising it to 3 costs 17 percentage points
    #: of recall and buys no precision (see docs/EVALUATION.md).
    min_cycles: float = 2.0


@dataclass
class PreprocessConfig:
    """Filtering and montage options applied before detection."""

    #: Mains frequency to notch out, plus harmonics up to Nyquist.
    #: ``None`` means "use the recording's", which each dataset carries (60 Hz
    #: for the US recordings, 50 Hz for Zurich). Set a number only to override
    #: it deliberately -- notching the wrong mains frequency leaves the
    #: interference in place *and* carves a hole where there was none, and it
    #: is the kind of mistake that never announces itself.
    line_freq: float | None = None
    notch: bool = True
    #: Bipolar re-referencing of neighbouring contacts on the same electrode.
    #: Standard practice for HFO work: it suppresses far-field and reference
    #: noise that a common reference shares across channels.
    bipolar: bool = True
    #: High-pass the continuous signal before anything else (removes drift).
    highpass: float = 1.0
    #: Low-pass, in Hz. ``None`` -- the default -- means none, which is what an
    #: HFO analysis wants: the band of interest runs to 500 Hz and anything
    #: that attenuates it is removing the signal. Exposed because a reviewer
    #: comparing against a conventional reading may want one, and refused by
    #: :func:`onset_hfo.preprocess.prepare` when it would cut into the band
    #: being analysed rather than applied quietly.
    lowpass: float | None = None
    #: Width of each notch, in Hz. Narrow on purpose: mains harmonics at
    #: 180/240 Hz sit inside the ripple band and a wide notch carves a hole in
    #: the signal being measured.
    notch_width: float = 2.0
    #: Notch the harmonics of the mains frequency as well as the fundamental.
    #: Off leaves 100/150/180/240 Hz interference inside the HFO bands.
    notch_harmonics: bool = True
    #: Re-reference to the average of all channels instead of to a neighbour.
    #: Only consulted when ``bipolar`` is off: the two are alternatives, and a
    #: common average re-introduces exactly the shared noise that the bipolar
    #: montage exists to suppress. Offered because it is standard practice
    #: elsewhere in EEG, with that caveat recorded in the steps.
    average_reference: bool = False
    #: Resample to this rate, in Hz. ``None`` keeps the recording's own.
    #: Downsampling is refused when it would put the analysed band above the
    #: new Nyquist -- the analysis would still run and the numbers would be
    #: meaningless.
    resample: float | None = None
    #: Drop channels flagged ``bad`` in the dataset's channels.tsv.
    drop_bads: bool = True
    #: Channels the reviewer marked bad themselves, on top of the dataset's.
    #: Carried here rather than on the recording so that the choice travels
    #: with the analysis configuration and lands in the report.
    exclude: tuple[str, ...] = ()


@dataclass
class PipelineConfig:
    """Everything the end-to-end pipeline needs, in one inspectable object."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    rms: DetectorConfig = field(default_factory=DetectorConfig)
    line_length: DetectorConfig = field(default_factory=lambda: DetectorConfig(threshold_sd=3.0))
    #: The two opt-in detectors. Their thresholds are **inherited** from the
    #: energy detector, not measured: no sweep has been run for either on real
    #: data. ``docs/EVALUATION.md`` §0 is what happens when an inherited
    #: default goes unchecked, so sweep before trusting these.
    hilbert: DetectorConfig = field(default_factory=DetectorConfig)
    short_time_energy: DetectorConfig = field(default_factory=DetectorConfig)
    spikes: SpikeConfig = field(default_factory=SpikeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    #: Seconds; the unit in which per-channel event rates are reported.
    rate_window_s: float = 60.0
    #: How many top channels the report lists.
    top_k: int = 5
    #: Rank difference at which two detectors are said to disagree.
    disagreement_ranks: int = 5

    def as_dict(self) -> dict:
        return {
            "preprocess": asdict(self.preprocess),
            "rms": asdict(self.rms),
            "line_length": asdict(self.line_length),
            "hilbert": asdict(self.hilbert),
            "short_time_energy": asdict(self.short_time_energy),
            "spikes": asdict(self.spikes),
            "validation": asdict(self.validation),
            "rate_window_s": self.rate_window_s,
            "top_k": self.top_k,
            "disagreement_ranks": self.disagreement_ranks,
        }


# --------------------------------------------------------------------------
# Measured operating points
# --------------------------------------------------------------------------

#: Detection thresholds, in robust SDs, with what each one is for.
#:
#: The shipped default (5.0) is Staba's published value and stays the default
#: because it is what a reviewer expects and what the ictal pipeline was built
#: with. But it is now *measured* rather than assumed, and on interictal data
#: it is a poor choice: scored against expert HFO markings on 20 subjects of
#: ds003498, 5.0 SD reaches recall 0.12 and ranks channels at Spearman 0.37,
#: against recall 0.38 and 0.66 at 2.0 SD. See ``docs/EVALUATION.md``.
#:
#: Use these by name rather than copying numbers around:
#:
#:     cfg = PipelineConfig()
#:     cfg.rms.threshold_sd = THRESHOLDS["interictal-agreement"]
THRESHOLDS: dict[str, float] = {
    # Staba et al. 2002, and this package's default.
    "literature": 5.0,
    # Highest agreement with expert channel ranking on ds003498 (rho 0.655).
    # Channel ranking is the clinically meaningful comparison: nobody operates
    # on an event, they operate on tissue.
    "interictal-agreement": 2.0,
    # Highest event-level F1 on the same cohort (0.418). Finds more events and
    # is wrong more often; prefer it when recall matters more than precision.
    "interictal-recall": 1.5,
    # The same criterion measured in the FAST RIPPLE band on the same cohort,
    # where the answer is 5.0 SD (rho 0.610, an interior maximum of a 2-10
    # sweep) -- numerically the literature value, but arrived at by
    # measurement and listed separately so the coincidence is not mistaken
    # for a default nobody checked.
    #
    # The gap between this and ``interictal-agreement`` is the practical
    # point: at 2.0 SD the fast-ripple detector runs at precision 0.086 and
    # returns ~1,140 detections per 60 s against a mean of 278 expert-marked
    # fast ripples per subject. One threshold for both bands is not a
    # simplification, it is a bug.
    #
    # That "278" was "~70" here until the Detectors page started reading
    # ``data/benchmark/`` instead of restating it: 70.6 is this detector's own
    # mean detection count at 5.0 SD, not the expert count, and the same
    # sentence in EVALUATION.md said 228. Three copies, two of them wrong;
    # ``data/benchmark/cohort.csv`` now carries the per-subject counts so the
    # number has a file behind it. See ``tests/test_app.py``.
    "interictal-agreement-fast-ripple": 5.0,
}

#: Version string stamped into every result table and report, so that a number
#: someone quotes months from now can be traced back to the code that made it.
PIPELINE_VERSION = "0.1.0"
