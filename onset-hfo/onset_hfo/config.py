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


#: Burstiness (envelope p99/p10) of a channel carrying no events at all.
#:
#: Band-pass Gaussian noise and its envelope is Rayleigh-distributed, whose
#: quantiles are ``sigma * sqrt(-2 ln(1 - p))`` -- so the ratio of the 99th
#: percentile to the 10th is a pure number, the same for a 2 uV contact and a
#: 200 uV one. That is what makes it usable as a floor without calibration:
#: a channel at this value has no event structure in it, whatever its gain.
RAYLEIGH_BURSTINESS: float = 6.611


@dataclass
class QualityConfig:
    """Which contacts and which stretches of time are fit to be analysed.

    Two kinds of verdict, and the difference is the most important thing in
    this class. A contact is **set aside** only for a fault where no
    physiology could produce the signal: flat, clipped at the amplifier's
    rail, swamped by mains, or with too little surviving time to rate. It is
    **flagged** when a measurement is unusual in a way that could equally be a
    fault or the finding -- and then it is analysed normally, and a human
    decides. Nothing in this software can tell a noisy amplifier from a
    contact full of real ripples by looking at band power, and the cost of
    guessing wrong is deleting the result.

    This stage exists because an HFO rate ranking is unusually easy to poison.
    A contact with a noisy amplifier produces ripple-band energy continuously;
    the detector finds it, the validator cannot tell it from signal because it
    *is* oscillatory, and that contact tops the ranking. A dead contact
    produces nothing and sits at the bottom looking reassuring. Neither is a
    statement about the brain, and neither announces itself in a table of
    rates.

    **Nothing here interpolates.** A standard M/EEG cleaner (``autoreject``,
    for one) repairs a bad channel from its neighbours. That is right when the
    quantity of interest is an evoked response averaged over sensors, and
    wrong here: the entire output of this software is a *per-channel* rate
    ranking, and an interpolated channel's rate is borrowed from the contacts
    beside it. It would read as a finding about that contact. So a bad channel
    is dropped and named, never repaired. (It is also moot on these datasets:
    interpolation needs contact coordinates and neither archive ships any.)

    Every threshold below is a floor for the gross cases, not a tuning knob
    that has been optimised -- no sweep has been run for any of them. They are
    deliberately loose, because the failure that matters here is rejecting
    real epileptic time, not keeping a little noise.
    """

    #: Reject a channel whose robust amplitude is below this, in microvolts:
    #: a disconnected or shorted contact. Real intracranial background sits
    #: well above it.
    flat_uv: float = 0.5
    #: Reject a channel that spends more than this fraction of the window
    #: pinned at its own extreme value -- an amplifier at its rail. The signal
    #: there is not small or noisy, it is absent, and a clipped edge rings
    #: through an 80-250 Hz filter like a textbook ripple.
    max_clipped_fraction: float = 0.01
    #: Reject a channel whose mains-frequency power (fundamental and
    #: harmonics) exceeds this fraction of its total power. Matters more here
    #: than in conventional EEG: the 4th and 5th harmonics of 50 Hz sit at
    #: 200 and 250 Hz, inside the ripple band being counted.
    max_line_fraction: float = 0.30
    #: **Flag** -- not reject -- a channel whose in-band-to-broadband power
    #: ratio is this many robust SDs above the montage's own median.
    #:
    #: A noisy amplifier produces continuous band-limited energy and tops an
    #: HFO ranking, which is the worst failure this software can have. This
    #: statistic finds it. It also finds the opposite: a contact full of real
    #: ripples has elevated band power *because the ripples are in the band*.
    #:
    #: It cannot tell them apart, and that was measured rather than assumed.
    #: On sub-13 of ds003498 this check flagged `TR1-TR2` and `TR2-TR3` at 10x
    #: and 6x the montage median -- and the archive's own annotators marked
    #: **91, 102 and 164** ripples on those three contacts. They are among the
    #: most epileptically active in the recording. Setting them aside would
    #: have blanked the finding, which is the same failure the segment test
    #: made and for the same reason.
    #:
    #: The quietest-second floor was tried as a discriminator, since real
    #: ripples are intermittent and an amplifier is not: a planted noisy
    #: contact scores 26x the montage's 10th percentile and `TR1-TR2` scores
    #: 5x. Better than the median, and still not a gap to put a threshold in.
    #:
    #: So this check measures and flags; it never removes. What it *can* do is
    #: say which way it leans -- see :attr:`bursty_ratio`.
    max_hf_ratio_sd: float = 6.0
    #: How many times the noise null a contact's envelope burstiness must
    #: reach before a band-power flag is described as activity rather than
    #: noise. Both are flags; neither removes anything. This only changes the
    #: sentence a reviewer reads, and the measured ratio is shown beside it.
    #:
    #: Burstiness is the 99th percentile of the ripple-band envelope over its
    #: 10th -- a within-channel dynamic range, so unlike the band-power ratio
    #: it carries no amplitude scale and needs no comparison to the montage.
    #: A contact full of ripples is tall spikes over a quiet floor; a noisy or
    #: poorly-coupled one is a raised carpet.
    #:
    #: **Its threshold is derived, not fitted.** For a channel whose
    #: band-passed signal is Gaussian noise the envelope is Rayleigh, so the
    #: ratio is a constant independent of amplitude:
    #: ``sqrt(-2 ln 0.01) / sqrt(-2 ln 0.90)`` = **6.611**
    #: (:data:`RAYLEIGH_BURSTINESS`; simulation through this project's own
    #: filter gives 6.65 +/- 0.08). A contact at that value is
    #: indistinguishable from filtered noise.
    #:
    #: The cohort agrees with the algebra. Across the ds003498 sweep, the 11
    #: distinct contacts that this stage flagged and the archive's annotators
    #: marked **not at all** score 6.6-6.9 -- the null, to within a rounding
    #: error. The 11 it flagged that they marked heavily score 8.2-60.6,
    #: median 29.4. No overlap.
    #:
    #: 1.2 sits in that gap, and deliberately near the bottom of it: calling a
    #: real contact noisy makes a reviewer under-weight a finding, which is
    #: the error this whole stage keeps making and keeps having to be stopped
    #: from. Erring toward "activity" is the safe side.
    #:
    #: Two limits, both of which are why this still only changes wording.
    #: The separation rests on 11 contacts against 11, from ten subjects of
    #: which only three carry both kinds, and is not validated out of sample.
    #: And burstiness separates *events* from *carpet*, not real from
    #: artifactual: an electrode popping once a second is bursty, and scores
    #: like a hippocampus full of ripples.
    bursty_ratio: float = 1.2
    #: Flag a channel whose overall amplitude is this many robust SDs from the
    #: montage's median, in either direction. Flagged for the same reason: a
    #: large-amplitude contact may be a gain fault or may be where the
    #: pathology is, and this cannot tell.
    max_amplitude_sd: float = 6.0
    #: The amplitude test needs this as well: the channel must differ from the
    #: montage median by at least this factor, not only by robust SDs.
    #:
    #: Because an outlier test needs a population and a montage is often a
    #: small one. On a fifteen-channel synthetic montage the robust SD across
    #: channels is small enough that an ordinary contact at 24 uV among
    #: neighbours at 36 clears 6 SD and would be thrown out -- a third of that
    #: montage was, before this was added. On a real 43-channel implantation
    #: nothing was.
    #:
    #: It guards the amplitude test **only**, and deliberately not the
    #: band-ratio one. The in-band share of power is not comparable between
    #: recordings the way amplitude is: a real intracranial contact puts about
    #: 0.05% of its power in the ripple band and this project's synthetic
    #: recording puts 13%, so a factor that is conservative on one is blind on
    #: the other -- it silenced the noisy-amplifier check entirely on the
    #: synthetic montage. That check keeps the robust-SD criterion alone,
    #: which is the one the quantity supports, at a strict 6 SD.
    amplitude_outlier_ratio: float = 3.0

    #: Length of the fixed segments the window is cut into, in seconds. The
    #: unit of time that can be rejected, and the resolution of the clean-time
    #: denominator each channel's rate is divided by.
    segment_s: float = 1.0
    #: Reject one channel's second when the *unfiltered* signal jumps by more
    #: than this many robust SDs of that channel's own sample-to-sample
    #: difference.
    #:
    #: **Discontinuity, not amplitude.** The first version of this stage
    #: rejected a segment whose peak-to-peak was 8 robust SDs above the
    #: channel's own median, which sounds conservative and is not: on sub-01
    #: of ds003498 it threw away six seconds of `AR2-AR3` -- the second
    #: busiest HFO channel in that window -- at 4-6x its median. Those
    #: deflections are 700-1000 uV on a bipolar depth contact, which is a
    #: textbook interictal discharge, not a fault. Amplitude alone cannot
    #: tell the pathology from the artifact, and the failure is silent and
    #: in the one direction that matters: it removes the epileptic seconds
    #: from the epileptic channel and lowers its rate.
    #:
    #: A jump can tell them apart, but not at the threshold the spike detector
    #: uses: :attr:`SpikeConfig.max_raw_jump_sd` is 10, and that is calibrated
    #: on a ~50 ms event window. The maximum of a heavy-tailed quantity grows
    #: with the number of samples it is taken over, and a 1 s segment at
    #: 2000 Hz has forty times as many, so 10 here rejected a tenth of the
    #: epileptic channels' seconds.
    #:
    #: So this one was measured directly. Over 20,520 channel-seconds from six
    #: windows of three ds003498 subjects, real intracranial seconds score a
    #: median of 3.7 SD, a 99th percentile of 10-16, and a maximum anywhere of
    #: **38**. Planted faults on the same data score **713** (saturation) and
    #: **1013** (an amplifier step). The gap is a factor of twenty, and 100
    #: sits in it: 2.6x above anything real that was measured, 7x below the
    #: mildest fault. A disconnection is caught by the flat test instead,
    #: which is what it is for.
    #:
    #: Its weakness, stated because it is not obvious: the statistic is
    #: relative to each channel's own sample-to-sample spread, so a recording
    #: with a lot of genuine high-frequency content has a larger denominator
    #: and a given fault scores lower against it. This project's own synthetic
    #: recording is such a case -- its planted transients reach 69 SD where
    #: real recordings reach 38 -- and a fault has to be about three times
    #: larger there before it trips. The absolute ceiling below is the
    #: backstop for that, and a fault milder than both is one a reviewer has
    #: to see on the trace. This stage is a floor for gross faults, not a
    #: guarantee of clean data.
    segment_jump_sd: float = 100.0
    #: And an absolute ceiling, in microvolts, for a segment no relative test
    #: would catch -- a window that is nothing but a pop. Set well above
    #: physiology: an intracranial discharge reaches a millivolt, so a lower
    #: ceiling would reproduce the mistake above with a constant.
    segment_ceiling_uv: float = 5000.0
    #: Drop a channel outright when this fraction of its segments is rejected.
    #: Below it the channel is kept and its rate divided by the time that
    #: survived; above it there is not enough left to call a rate.
    max_bad_segment_fraction: float = 0.5


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

    # -- filter design ---------------------------------------------------------
    #: ``"fir"`` (the default: a windowed zero-phase FIR, what MNE and this
    #: project have always used) or ``"iir"`` (a Butterworth, applied forwards
    #: and backwards for zero phase). HFO detection is filter-sensitive: a
    #: long FIR rings at a sharp discharge and the ringing looks like a
    #: ripple, which is why the choice is exposed rather than fixed.
    filter_method: str = "fir"
    #: Butterworth order for the IIR design. Applied twice (zero phase), so
    #: the effective order is double this.
    iir_order: int = 4
    #: ``"zero"`` (no delay, the default) or ``"minimum"`` (causal FIR: a
    #: ripple is never smeared backwards in time, at the cost of a delay).
    filter_phase: str = "zero"
    #: FIR transition bandwidth in Hz for the high- and low-pass edges;
    #: ``None`` lets MNE choose from the cut-off. Narrower is sharper and
    #: longer; longer rings more.
    transition_bandwidth: float | None = None

    # -- reference ------------------------------------------------------------
    #: The reference scheme by name: ``"bipolar"``, ``"average"``,
    #: ``"median"`` (common median: the average's robust cousin, which one
    #: faulty contact cannot drag), ``"shaft"`` (each contact minus the mean
    #: of its own electrode shaft, the usual choice for SEEG) or ``"none"``.
    #: ``None`` keeps the older two flags deciding, so saved configurations
    #: read as they did.
    reference: str | None = None

    # -- artifact annotation ----------------------------------------------------
    #: Mark seconds where broadband high-frequency power rises across the
    #: montage at once, as muscle and movement do (MNE's
    #: ``annotate_muscle_zscore``). The marked seconds are set aside by the
    #: data-quality stage. Off by default: its band overlaps the ripple band,
    #: and a reviewer should turn it on knowing that.
    annotate_muscle: bool = False
    #: The z-score above which a second is marked muscle.
    muscle_z: float = 4.0
    #: Mark seconds whose peak-to-peak amplitude exceeds a ceiling, per
    #: contact (MNE's ``annotate_amplitude``). Off by default for the reason
    #: :class:`QualityConfig` documents: amplitude cannot tell a discharge
    #: from a pop, and the seconds it removes are the epileptic ones.
    annotate_amplitude: bool = False
    #: The peak-to-peak ceiling in microvolts; ``None`` learns one per contact
    #: from the data by cross-validation, the way ``autoreject`` does for its
    #: global threshold -- and, like it, learns a ceiling that discharges
    #: exceed. Nothing is interpolated.
    amplitude_ptp_uv: float | None = None


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
    quality: QualityConfig = field(default_factory=QualityConfig)
    #: Run the data-quality stage at all. On by default: a rate ranking with
    #: a noisy amplifier at the top of it is the failure this project is most
    #: likely to produce, and the stage costs a fraction of a second. Off
    #: reproduces every number this project measured before the stage existed,
    #: which is why it is a switch rather than a removal.
    check_quality: bool = True
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
            "quality": asdict(self.quality),
            "check_quality": self.check_quality,
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
