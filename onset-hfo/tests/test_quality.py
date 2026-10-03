"""Which contacts and which seconds are fit to analyse.

This stage is the one most able to do quiet damage, so most of these tests are
about what it must **not** do. An HFO rate ranking is driven by a handful of
channels and a handful of seconds; a cleaner that removes the epileptic ones
lowers exactly the rate that matters and reports nothing unusual, and the
output still looks like a finding.

The history is in `onset_hfo.config.QualityConfig.segment_jump_sd`, and it is
why `test_a_discharge_is_not_an_artifact` exists: the first version of the
segment test rejected a segment whose peak-to-peak was 8 robust SDs above the
channel's own median, and on sub-01 of ds003498 that threw away six seconds of
`AR2-AR3`, the second busiest HFO channel in the window. An interictal
discharge is a large deflection. Amplitude cannot tell it from a fault.

Everything here runs on the synthetic recording with faults planted into it,
so none of it needs a cached slice or a network. The thresholds themselves
were measured on real recordings; the measurement is recorded in the config,
and what is pinned here is the separation it depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_hfo.config import QualityConfig
from onset_hfo.preprocess import prepare
from onset_hfo.quality import (
    REASONS,
    SET_ASIDE,
    channel_quality,
    clean_seconds,
    quality_summary,
    segment_quality,
    usable_channels,
)


@pytest.fixture(scope="module")
def prep(recording):
    return prepare(recording, verbose=False)


def _with_fault(prep, channel, fault):
    """A copy of `prep` with one channel replaced by `fault(signal, t)`."""
    import copy

    out = copy.deepcopy(prep)
    i = out.ch_names.index(channel)
    t = np.arange(out.data.shape[1]) / out.sfreq
    out.data[i] = fault(out.data[i], t)
    return out


# -- the channel checks ----------------------------------------------------

def test_a_clean_montage_passes_every_channel(prep):
    """The baseline that makes every rejection below mean something."""
    quality = channel_quality(prep)
    assert len(quality) == prep.n_channels
    assert quality["good"].all(), list(quality[~quality["good"]]["channel"])
    assert usable_channels(quality) == list(prep.ch_names)
    assert "All" in quality_summary(quality)


@pytest.mark.parametrize("label,reason,fault", [
    ("a dead or disconnected contact", "flat",
     lambda x, t: np.random.default_rng(0).normal(0, 0.05, x.size)),
    ("an amplifier sitting at its rail", "clipped",
     lambda x, t: np.clip(x, -np.percentile(np.abs(x), 60),
                          np.percentile(np.abs(x), 60))),
    ("mains interference swamping the contact", "line_noise",
     lambda x, t: x + 50 * np.std(x) * np.sin(2 * np.pi * 60 * t)),
    ("a noisy amplifier", "hf_noise",
     lambda x, t: x + np.random.default_rng(1).normal(0, 6 * np.std(x), x.size)),
])
def test_each_channel_fault_is_caught_and_named(prep, label, reason, fault):
    """One reason per fault, and no collateral damage to the rest.

    The second half matters as much as the first: these tests compare each
    channel against the montage's own distribution, so a check that flags one
    broken contact by dragging its neighbours out with it has not worked.
    """
    broken = prep.ch_names[0]
    quality = channel_quality(_with_fault(prep, broken, fault))
    row = quality[quality["channel"] == broken].iloc[0]
    assert row["reason"] == reason, f"{label}: {row['reason']} != {reason}"
    # Set aside for the unambiguous faults; analysed and flagged for the ones
    # that could equally be the finding.
    # `bool(...)`: pandas hands back numpy booleans, which are not `True`.
    assert bool(row["good"]) is (reason not in SET_ASIDE)
    assert bool(row["flagged"]) is (reason not in SET_ASIDE)

    others = quality[quality["channel"] != broken]
    collateral = others[(~others["good"]) | others["flagged"]]
    assert collateral.empty, f"{label} also caught {list(collateral['channel'])}"


def test_a_noisy_amplifier_is_flagged_and_still_analysed(prep):
    """Why the in-band ratio is measured, and why it must not remove anything.

    A contact with a noisy amplifier does not look broken in a rate table: it
    looks like the finding. Its noise is genuinely oscillatory, so
    `onset_hfo.validate` cannot throw the events out either -- its checks ask
    whether a narrow-band oscillation is present, and one is. This is the only
    stage that can see it at all.

    And it cannot tell it from the real thing. A contact full of ripples has
    elevated band power *because the ripples are in the band*. On sub-13 of
    ds003498 this check flagged `TR1-TR2` and `TR2-TR3`, and the archive's own
    annotators marked 91, 102 and 164 ripples on those three contacts. Setting
    them aside would have deleted the finding.

    So the contract this test exists to hold is: measure it, say it loudly,
    analyse the channel anyway, and leave the judgement to someone who can
    look at the trace.
    """
    broken = prep.ch_names[0]
    noisy = _with_fault(
        prep, broken,
        lambda x, t: x + np.random.default_rng(2).normal(0, 6 * np.std(x), x.size))
    quality = channel_quality(noisy)
    row = quality[quality["channel"] == broken].iloc[0]

    assert row["reason"] == "hf_noise"
    assert bool(row["flagged"]) is True
    assert bool(row["good"]) is True, "a flagged contact must still be analysed"
    assert "hf_noise" not in SET_ASIDE
    assert broken in usable_channels(quality)

    # It is an outlier upward in band power, not merely different.
    assert row["hf_ratio"] == quality["hf_ratio"].max()
    assert row["hf_ratio_sd"] > quality.drop(index=row.name)["hf_ratio_sd"].max()

    # And the summary tells the reviewer what to do about it.
    text = quality_summary(quality)
    assert "analysed but flagged" in text
    assert broken in text
    assert "look at these on the trace" in text


def test_a_flagged_contact_keeps_its_rate_and_its_time(prep):
    """The behavioural half of the same contract, where it would be felt.

    A flagged contact that lost its clean time would have a blank rate, which
    is removal by another route.
    """
    from onset_hfo.quality import analysable_seconds

    broken = prep.ch_names[0]
    noisy = _with_fault(
        prep, broken,
        lambda x, t: x + np.random.default_rng(3).normal(0, 6 * np.std(x), x.size))
    segments = segment_quality(noisy)
    quality = channel_quality(noisy, segments=segments)
    assert quality[quality["channel"] == broken].iloc[0]["flagged"]

    seconds = analysable_seconds(segments, quality, list(prep.ch_names),
                                 prep.duration)
    assert seconds[broken] > 0.0


def test_every_reason_the_code_can_give_has_a_sentence():
    """A verdict a reviewer cannot act on is not a verdict."""
    emitted = {"flat", "clipped", "line_noise", "hf_noise", "amplitude",
               "mostly_bad_segments", "segment_jump", "segment_ceiling",
               "segment_flat"}
    assert emitted <= set(REASONS)
    for reason, sentence in REASONS.items():
        # A sentence a reviewer can act on, not a restatement of the label.
        assert len(sentence.split()) >= 8, reason


def test_a_reviewer_can_reinstate_a_channel_the_checks_rejected(prep):
    """The machine proposes. Overriding is one-directional on purpose.

    Putting a channel back is a reviewer saying they looked at it. Taking one
    out is already possible through `PreprocessConfig.exclude`, where it is
    recorded as their choice rather than buried as an override of a verdict.
    """
    broken = prep.ch_names[0]
    quality = channel_quality(
        _with_fault(prep, broken, lambda x, t: np.zeros_like(x)))
    assert broken not in usable_channels(quality)
    assert broken in usable_channels(quality, keep=(broken,))
    assert "reinstated by the reviewer" in quality_summary(quality, kept=(broken,))


# -- the segment checks ----------------------------------------------------

def test_a_discharge_is_not_an_artifact(prep):
    """The test this stage exists to pass, and once failed.

    An interictal discharge is a large, sharp deflection -- the largest thing
    in a normal intracranial second. A segment test that rejects on amplitude
    throws away the seconds that carry the pathology, on the channels that
    carry it, and lowers precisely the rates a reviewer is reading. It does so
    silently, and the ranking it produces still looks like a result.

    So: a modelled discharge, at a hundred times the background and with the
    rise time of a real one, must survive.
    """
    channel = prep.ch_names[0]

    def discharge(x, t):
        """Three 20 ms discharges of 800 µV — large, and entirely real."""
        out = x.copy()
        for at in (2.0, 5.0, 9.0):
            centre = int(at * prep.sfreq)
            width = int(0.020 * prep.sfreq)
            out[centre - width:centre + width] += 800.0 * np.hanning(2 * width)
        return out

    faulted = _with_fault(prep, channel, discharge)
    row = faulted.data[faulted.ch_names.index(channel)]
    assert np.ptp(row) < QualityConfig().segment_ceiling_uv, (
        "the modelled discharge is beyond physiology, so this would be "
        "testing the ceiling rather than the discrimination")

    segments = segment_quality(faulted)
    mine = segments[segments["channel"] == channel]
    assert mine["good"].all(), (
        "a modelled discharge was rejected as an artifact:\n"
        + mine[~mine["good"]][["t_start", "ptp_uv", "jump_sd"]].to_string())


@pytest.mark.parametrize("label,reason,fault", [
    # Smaller than this recording's own peak-to-peak, so nothing about its
    # amplitude is remarkable; what gives it away is arriving in one sample,
    # which no physiology does. That is the case a peak-to-peak test cannot
    # see and this one must. It takes 3 mV here and would take about 1 mV on
    # a real recording -- the synthetic fixture's own planted transients
    # inflate the denominator this is measured against, which the config
    # records as the limitation it is.
    ("an amplifier step", "segment_jump",
     lambda x, t: x + np.where((t >= 2.0) & (t < 2.2), 3000.0, 0.0)),
    ("a disconnection", "segment_flat",
     lambda x, t: np.where((t >= 2.0) & (t < 3.0), 0.0, x)),
])
def test_each_segment_fault_is_caught_and_named(prep, label, reason, fault):
    channel = prep.ch_names[0]
    segments = segment_quality(_with_fault(prep, channel, fault))
    mine = segments[segments["channel"] == channel]
    bad = mine[~mine["good"]]
    assert not bad.empty, f"{label} was not caught"
    assert set(bad["reason"]) == {reason}
    # The fault is one or two seconds long; it must not take the window with it.
    assert len(bad) <= 3, f"{label} rejected {len(bad)} segments"


def test_a_step_scores_an_order_of_magnitude_above_anything_physiological(prep):
    """The separation the default threshold sits in, pinned.

    Measured on real recordings (see `QualityConfig.segment_jump_sd`): real
    intracranial seconds reach 38 SD at the very worst, planted faults score
    700-1000, and the default of 100 sits in that gap.

    This fixture is the harder case, and deliberately so. The synthetic
    recording plants sharp transients of its own, so its clean seconds reach
    about 69 SD and its denominator is inflated -- the margin here is
    therefore much tighter than on real data. If a change to the statistic
    narrows it further, this is where it should surface, rather than in a
    reviewer's rate table.
    """
    channel = prep.ch_names[0]
    clean = segment_quality(prep)
    worst_clean = float(clean["jump_sd"].max())

    stepped = _with_fault(
        prep, channel,
        lambda x, t: x + np.where(t >= 2.0, 3000.0, 0.0))
    faulted = segment_quality(stepped)
    step = float(faulted[(faulted["channel"] == channel)
                         & (faulted["segment"] == 2)]["jump_sd"].iloc[0])

    assert worst_clean < QualityConfig().segment_jump_sd < step, (
        f"clean {worst_clean:.0f} / threshold "
        f"{QualityConfig().segment_jump_sd:.0f} / step {step:.0f}")


def test_the_segment_grid_covers_the_window_in_original_recording_times(prep):
    segments = segment_quality(prep)
    per_channel = len(segments) // prep.n_channels
    assert per_channel == int(prep.duration // QualityConfig().segment_s)
    first = segments.iloc[0]
    assert first["t_start"] == pytest.approx(prep.t_offset)
    assert first["t_stop"] - first["t_start"] == pytest.approx(
        QualityConfig().segment_s)


def test_a_window_shorter_than_one_segment_is_empty_rather_than_an_error(prep):
    import copy

    tiny = copy.deepcopy(prep)
    tiny.data = tiny.data[:, :10]
    assert segment_quality(tiny).empty
    assert clean_seconds(segment_quality(tiny)) == {}


# -- the denominator -------------------------------------------------------

def test_clean_time_is_counted_per_channel(prep):
    """The number that makes rejection honest rather than cosmetic.

    Rejecting four seconds of one channel and then dividing every channel by
    the window's nominal length reports a rate per minute of a minute that was
    not analysed, and the Poisson interval inherits the error. The rejection
    would make the table *less* accurate than leaving the artifact in.
    """
    channel = prep.ch_names[0]
    segments = segment_quality(_with_fault(
        prep, channel,
        lambda x, t: np.where((t >= 2.0) & (t < 5.0), 0.0, x)))

    seconds = clean_seconds(segments, list(prep.ch_names))
    assert seconds[channel] == pytest.approx(prep.duration - 3.0, abs=1.0)
    others = [v for k, v in seconds.items() if k != channel]
    assert all(v == pytest.approx(max(others)) for v in others)
    assert seconds[channel] < max(others)


def test_clean_time_falls_back_to_the_whole_window_when_nothing_was_checked(prep):
    """Turning the stage off must not silently zero every denominator."""
    assert clean_seconds(pd.DataFrame(), list(prep.ch_names), prep.duration) == {
        name: prep.duration for name in prep.ch_names}


def test_a_channel_that_is_mostly_rejected_is_dropped_rather_than_rated(prep):
    """Below some amount of surviving time a rate is not a measurement.

    This is the residual verdict: it fires for a channel that passes all five
    per-channel checks and still loses most of its seconds. That is rare by
    construction -- a fault large enough to reject segment after segment is
    usually large enough to show up in the channel's own amplitude or band
    ratio first, and when it does the more specific reason is the more useful
    one. So the rule is exercised directly, on a segment table, rather than
    through a signal contrived to slip past the other five.
    """
    channel = prep.ch_names[0]
    segments = segment_quality(prep)
    mostly_bad = segments["channel"] == channel
    segments.loc[mostly_bad & (segments["segment"] % 4 != 0), "good"] = False

    quality = channel_quality(prep, segments=segments)
    row = quality[quality["channel"] == channel].iloc[0]
    assert row["bad_segment_fraction"] > QualityConfig().max_bad_segment_fraction
    assert not row["good"]
    assert row["reason"] == "mostly_bad_segments"
    assert channel not in usable_channels(quality)

    # And the same channel, with its segments left alone, is fine. Otherwise
    # this would pass on a bug that rejects everything.
    assert channel_quality(prep, segments=segment_quality(prep)).iloc[0]["good"]


def test_a_channel_popping_once_a_second_is_caught_before_that(prep):
    """The realistic version, and why the residual rule is rarely reached.

    A contact that pops briefly every second loses every segment -- and is
    also, measurably, a noisy channel. The band-ratio check sees it first and
    says something more useful than "not enough clean time".
    """
    def popping(x, t):
        out = x.copy()
        width = int(0.003 * prep.sfreq)
        for second in range(int(prep.duration)):
            at = int((second + 0.5) * prep.sfreq)
            out[at:at + width] += 3000.0
        return out

    broken = _with_fault(prep, prep.ch_names[0], popping)
    segments = segment_quality(broken)
    mine = segments[segments["channel"] == prep.ch_names[0]]
    assert not mine["good"].any()           # every second rejected
    assert channel_quality(broken, segments=segments).iloc[0]["reason"] == "hf_noise"


def test_the_summary_says_what_was_set_aside_and_what_it_costs(prep):
    channel = prep.ch_names[0]
    broken = _with_fault(prep, channel, lambda x, t: np.zeros_like(x))
    segments = segment_quality(broken)
    text = quality_summary(channel_quality(broken, segments=segments), segments)
    assert "set aside" in text
    assert "rate is over the time that survived" in text


def test_the_quality_module_needs_no_qt():
    import ast
    import pathlib

    import onset_hfo.quality as target

    tree = ast.parse(pathlib.Path(target.__file__).read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for name in names:
            assert name.split(".")[0] not in {"qtpy", "PySide6", "PyQt5",
                                              "PyQt6", "onset_review"}, name


def test_only_unambiguous_faults_remove_a_contact():
    """The most consequential list in this module, pinned as a list.

    Every reason here says the signal is absent or corrupted in a way no
    physiology produces, so there is nothing for a human to adjudicate.
    Everything else is a measurement that is as consistent with the finding as
    with a fault, and adding one to this tuple means deciding to delete
    findings. `QualityConfig.max_hf_ratio_sd` records the recording that
    settled where the line goes.
    """
    assert set(SET_ASIDE) == {"flat", "clipped", "line_noise",
                              "mostly_bad_segments"}
    assert "hf_noise" not in SET_ASIDE
    assert "amplitude" not in SET_ASIDE
    for reason in SET_ASIDE:
        assert reason in REASONS
