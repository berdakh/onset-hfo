"""The orchestration half: labels, contract, evidence store, live tools,
the S0-S3 ladder, the two verifiers, and SOZ scoring.

Everything here is offline. No download, no model weights, no GPU: the
language-model paths are exercised against small fake backends that reply the
way a served Qwen or Llama does, including the ways they get it wrong.
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd
import pytest

from onset_agent.analysis import AnalysisSession, build_registry
from onset_agent.backends import AssistantMessage, Backend
from onset_agent.contract import ToolRun
from onset_agent.evidence import EvidenceStore
from onset_agent.planner import (
    FIXED_PLAN,
    FixedBudget,
    ModelJudged,
    Rung,
    ScriptedPlanner,
    TiedSetWidth,
    rank_channels,
    run_rung,
)
from onset_agent.scoring import score_ranking
from onset_agent.verifier import (
    DeterministicVerifier,
    LLMVerifier,
    claim_numbers,
    compare_verifiers,
    split_sentences,
)
from onset_hfo.cohort import (
    SozLabels,
    decode_outcome,
    expand_contacts,
    label_channels,
    labels_from_ground_truth,
    load_labels_csv,
    normalize_subject,
    placeholder_labels,
    soz_labels,
    write_label_template,
)

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_expand_contacts_handles_the_clinicians_shorthand():
    assert expand_contacts("TT1-3; AST1, mst2") == {"TT1", "TT2", "TT3", "AST1", "MST2"}
    assert expand_contacts("PST1-4: AST1-2, MST1-2") == {
        "PST1", "PST2", "PST3", "PST4", "AST1", "AST2", "MST1", "MST2"}
    # A trailing newline and an empty token must not produce a phantom contact.
    assert expand_contacts("G4, G5,\n") == {"G4", "G5"}
    assert expand_contacts(None) == set()
    assert expand_contacts(float("nan")) == set()


def test_expand_contacts_refuses_an_absurd_range():
    """A range wider than an electrode is a parsing accident, not 900 contacts."""
    assert expand_contacts("A1-9999") == set()


def test_normalize_subject_reconciles_archive_and_spreadsheet_ids():
    # The S3 tree says sub-pt01; the spreadsheet says pt1. Both must agree.
    assert normalize_subject("sub-pt01") == normalize_subject("pt1") == "pt1"
    assert normalize_subject("sub-umf001") == normalize_subject("umf001") == "umf1"
    assert normalize_subject("sub-ummc002") == "ummc2"
    assert normalize_subject("jh101") == "jh101"


def test_outcome_code_s_means_seizure_free():
    """The trap: 'F' is failure, not free. Inverting this inverts every label."""
    assert decode_outcome("S") is True
    assert decode_outcome("F") is False
    assert decode_outcome("NR") is None
    assert decode_outcome(None) is None


def test_a_bipolar_channel_is_soz_when_either_contact_is():
    labels = SozLabels(subject="x", soz_contacts={"AD1", "ATT1"})
    frame = label_channels(["AD1-AD2", "AD3-AD4", "ATT1-ATT2"], labels)
    assert frame.set_index("channel")["is_soz"].to_dict() == {
        "AD1-AD2": True, "AD3-AD4": False, "ATT1-ATT2": True}


def test_trustworthy_requires_a_curated_label_and_a_working_surgery():
    curated = SozLabels("x", {"A1"}, source="clinical_summary", seizure_free=True)
    failed = SozLabels("x", {"A1"}, source="clinical_summary", seizure_free=False)
    weak = SozLabels("x", {"A1"}, source="events_markers", seizure_free=True)
    assert curated.trustworthy and not failed.trustworthy and not weak.trustworthy


def test_local_label_csv_is_a_drop_in_replacement(tmp_path):
    path = tmp_path / "soz.csv"
    path.write_text("subject,soz_contacts,engel,seizure_free,site\n"
                    "anon-01,\"LA1-3; LH2\",1,True,our-centre\n")
    labels = load_labels_csv(path)["anon-01"]
    assert labels.soz_contacts == {"LA1", "LA2", "LA3", "LH2"}
    assert labels.source == "local" and labels.engel == 1


def test_a_label_csv_without_the_required_columns_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("subject,contacts\nanon-01,LA1\n")
    with pytest.raises(ValueError, match="soz_contacts"):
        load_labels_csv(path)


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def test_unknown_tools_and_arguments_are_refused_before_anything_runs(registry):
    assert not registry.run("no_such_tool").ok
    run = registry.run("detect_hfo", {"nonsense": 1})
    assert not run.ok and "does not accept nonsense" in run.error


def test_a_wrong_type_is_refused_but_an_out_of_range_value_is_clamped(registry):
    assert not registry.run("detect_hfo", {"detector": 5}).ok
    assert not registry.run("detect_hfo", {"detector": "wavelet"}).ok
    # Clamping, not rejecting: a planner exploring parameters is corrected,
    # and the record shows what actually ran.
    run = registry.run("detect_hfo", {"threshold_sd": 99, "k": 2})
    assert run.ok and run.input["threshold_sd"] == 12.0


def test_a_handler_error_meant_for_the_model_is_passed_through(registry):
    run = registry.run("detect_hfo", {"channels": ["NOSUCH1-NOSUCH2"]})
    assert not run.ok
    assert "unknown channel" in run.error and "channel_qc" in run.error


def test_run_ids_are_sequential_and_name_their_tool(registry):
    assert registry.run("get_recording_metadata").run_id == "meta_001"
    assert registry.run("get_recording_metadata").run_id == "meta_002"
    assert registry.run("channel_qc").run_id == "qc_001"


def test_a_run_record_has_the_shape_the_contract_promises(registry):
    record = registry.run("detect_hfo", {"k": 2}).as_record()
    assert set(record) == {"tool", "input", "output", "run_id", "runtime_s"}


# --------------------------------------------------------------------------
# The evidence store
# --------------------------------------------------------------------------


def test_a_number_resolves_to_the_run_that_produced_it():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {}, {"A-B": {"rate_per_min": 14.23}}))
    assert store.find_number(14.23) == ["hfo_001"]
    assert store.find_number(14.2) == ["hfo_001"]      # honest rounding
    assert store.find_number(15.5) == []               # not a rounding of anything
    assert store.find_number(1423) == []


def test_a_failed_run_is_still_recorded_but_contributes_no_numbers():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {}, {"error": "boom"}, ok=False,
                         error="boom"))
    assert len(store) == 1 and store.cost()["n_failed"] == 1
    assert store.all_numbers() == set()


def test_duplicate_run_ids_are_a_programming_error():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {}, {}))
    with pytest.raises(ValueError, match="already in the store"):
        store.append(ToolRun("hfo_001", "detect_hfo", {}, {}))


def test_the_digest_keeps_the_ranking_the_tool_computed():
    """Sorting the channel keys would hand the planner a different ranking."""
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {},
                         {"channels": {"ZZ1-ZZ2": {"rate_per_min": 90.0},
                                       "AA1-AA2": {"rate_per_min": 1.0}}}))
    assert store.digest().index("ZZ1-ZZ2") < store.digest().index("AA1-AA2")


# --------------------------------------------------------------------------
# Live tools
# --------------------------------------------------------------------------


def test_a_stricter_threshold_finds_fewer_events(registry):
    loose = registry.run("detect_hfo", {"threshold_sd": 4, "k": 5})
    strict = registry.run("detect_hfo", {"threshold_sd": 8, "k": 5})
    assert loose.output["n_accepted"] > strict.output["n_accepted"]


def test_the_same_call_twice_gives_the_same_numbers_and_is_cached(registry):
    first = registry.run("detect_hfo", {"threshold_sd": 5, "k": 3})
    second = registry.run("detect_hfo", {"threshold_sd": 5, "k": 3})
    assert first.output["channels"] == second.output["channels"]
    assert second.output["cached"] is True and first.run_id != second.run_id


def test_a_window_restricts_the_analysis_to_that_time(registry):
    run = registry.run("detect_hfo", {"window_s": [10, 20], "k": 3})
    assert run.ok and run.output["window_s"] == [10.0, 20.0]
    evidence = registry.run("get_event_evidence",
                            {"channel": run.output["channels"] and
                             next(iter(run.output["channels"])), "k": 1})
    assert evidence.ok


def test_an_unusably_high_band_is_refused_with_the_reason(recording):
    """Fast ripples against the Nyquist edge are not measurable, and saying so
    is better than returning numbers nobody should trust."""
    from dataclasses import replace as dc_replace

    slow = dc_replace(recording)
    slow.raw = recording.raw.copy().resample(1000)
    run = build_registry(AnalysisSession(slow)).run("detect_hfo", {"band": "fast_ripple"})
    assert not run.ok and "Nyquist" in run.error


def test_every_tool_returns_json_serialisable_output(registry, session):
    top = next(iter(registry.run("detect_hfo", {"k": 2}).output["channels"]))
    calls = [("get_recording_metadata", {}), ("channel_qc", {"max_channels": 2}),
             ("detect_spikes", {"k": 2}), ("spectral_power", {"channels": [top]}),
             ("rate_change", {"k": 2}), ("compare_detectors", {}),
             ("propagation_lead", {"k": 2}), ("get_event_evidence", {"channel": top})]
    for name, args in calls:
        run = registry.run(name, args)
        assert run.ok, f"{name} failed: {run.error}"
        json.dumps(run.as_record(), default=str)


def test_no_tool_can_change_which_patient_is_analysed(registry):
    for spec in registry.specs.values():
        assert "subject" not in spec.parameters["properties"]
        assert "patient" not in spec.parameters["properties"]


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def _store_with(rates):
    store = EvidenceStore()
    for i, (threshold, channels) in enumerate(rates, start=1):
        store.append(ToolRun(f"hfo_{i:03d}", "detect_hfo", {},
                             {"threshold_sd": threshold,
                              "channels": {c: {"rate_per_min": r} for c, r in channels}}))
    return store


def test_a_channel_that_collapses_under_scrutiny_is_demoted():
    store = _store_with([(5.0, [("A-B", 100.0), ("C-D", 80.0)]),
                         (7.0, [("A-B", 10.0)])])
    ranked = {r.channel: r for r in rank_channels(store)}
    assert ranked["A-B"].robustness == pytest.approx(0.1)
    assert ranked["C-D"].robustness == 1.0
    assert [r.channel for r in rank_channels(store)] == ["C-D", "A-B"]


def test_a_retest_can_never_promote_a_channel():
    """Robustness is capped at 1: scrutiny lowers confidence, never raises it."""
    store = _store_with([(5.0, [("A-B", 10.0)]), (7.0, [("A-B", 999.0)])])
    assert rank_channels(store)[0].robustness == 1.0


def test_an_unchallenged_channel_is_unchallenged_not_verified():
    store = _store_with([(5.0, [("A-B", 10.0)])])
    ranked = rank_channels(store)[0]
    assert ranked.robustness == 1.0 and ranked.retested_at == []


def test_the_fixed_plan_never_retests_so_the_multiplier_cannot_reach_S0():
    """The premise of the multiplier ablation, pinned structurally.

    `docs/EVALUATION.md` §6c measures the multiplier as a ranking rule rather
    than through the ladder, and that is only a fair account of S0 if S0 never
    re-tests. It does not: `FIXED_PLAN` calls `detect_hfo` once and sets no
    threshold, so `rank_channels` sees one threshold per channel, the stricter
    loop never runs, and score reduces to the survey rate exactly.
    """
    detects = [params for name, params in FIXED_PLAN if name == "detect_hfo"]
    assert len(detects) == 1, (
        f"FIXED_PLAN now makes {len(detects)} detect_hfo calls; if any two use "
        f"different thresholds the multiplier reaches S0 and §6c needs redoing")
    assert "threshold_sd" not in detects[0]

    # And with one threshold the rule is inert, not merely usually inert.
    store = _store_with([(2.0, [("A-B", 10.0), ("C-D", 4.0)])])
    for ranked in rank_channels(store):
        assert ranked.robustness == 1.0
        assert ranked.score == pytest.approx(ranked.survey_rate_per_min)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_the_fixed_pipeline_never_challenges_itself(session):
    session.reset_memo()
    result = run_rung(session, Rung.S0)
    assert result.n_retested == 0
    assert result.ranking and result.backend.startswith("none")
    assert "no mechanism for challenging" in result.report


def test_single_shot_is_never_shown_its_results(session):
    """S1 must not be able to re-plan: that is what makes it the S1 rung."""
    session.reset_memo()
    result = run_rung(session, Rung.S1, stop_rule=ModelJudged(max_calls=8))
    assert result.n_retested == 0


def test_replanning_challenges_the_leaders_and_can_reorder_them(session):
    session.reset_memo()
    fixed = run_rung(session, Rung.S0)
    session.reset_memo()
    planned = run_rung(session, Rung.S2, stop_rule=ModelJudged(max_calls=12))
    assert planned.n_retested > 0
    challenged = {r.channel for r in planned.ranking if r.retested_at}
    assert challenged <= set(fixed.top_k(5)), "re-tests should target the leaders"


def test_every_rung_uses_the_same_analyzers(session):
    for rung in Rung:
        session.reset_memo()
        result = run_rung(session, rung, stop_rule=FixedBudget(k=5))
        for step in result.steps:
            if step.get("type") == "tool":
                assert step["tool"] in build_registry(session).names()


def test_the_stopping_rules_stop_for_different_reasons(session):
    session.reset_memo()
    budget = run_rung(session, Rung.S2, stop_rule=FixedBudget(k=3))
    assert budget.cost["n_tool_calls"] == 3 and "budget" in budget.stop_reason
    session.reset_memo()
    tied = run_rung(session, Rung.S2, stop_rule=TiedSetWidth(max_width=2, max_calls=12))
    # On this recording many channels are statistically tied, so the width
    # rule never fires and the planner runs out of things to try. That is a
    # legitimate outcome and must be reported as one, not as a step limit.
    assert "tied" in tied.stop_reason or "nothing further" in tied.stop_reason
    assert "step limit" not in tied.stop_reason


def test_the_tied_set_rule_does_not_claim_a_coverage_guarantee():
    """It is a tied-rank set from Poisson intervals, not a conformal set."""
    assert TiedSetWidth().guarantees_coverage is False


def test_a_malformed_plan_does_not_end_the_session(session):
    class Babbling(Backend):
        name = "babbling"

        def chat(self, messages, tools):
            return AssistantMessage(content="Sure! Let me think about that.")

    session.reset_memo()
    result = run_rung(session, Rung.S2, backend=Babbling(), max_steps=3)
    assert any(s["type"] == "format_error" for s in result.steps)
    assert result.ranking == []


def test_the_audit_trail_survives_a_round_trip(session, tmp_path):
    session.reset_memo()
    result = run_rung(session, Rung.S2, stop_rule=FixedBudget(k=4))
    path = result.store.save(tmp_path / "evidence.json")
    payload = json.loads(path.read_text())
    assert payload["contract_version"] and len(payload["runs"]) == 4
    assert all("run_id" in r for r in payload["runs"])


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@pytest.fixture
def verified_store():
    store = EvidenceStore()
    store.append(ToolRun("hfo_001", "detect_hfo", {"threshold_sd": 5},
                         {"channels": {"A-B": {"rate_per_min": 14.2}}}))
    store.append(ToolRun("compare_001", "compare_detectors", {},
                         {"agreement": {"jaccard": 0.53}}))
    return store


def test_sentences_split_without_breaking_decimals():
    assert split_sentences("A had 14.2 ripples/min [hfo_001]. B was quiet.") == [
        "A had 14.2 ripples/min [hfo_001].", "B was quiet."]


def test_channel_names_and_small_counts_are_not_claims():
    assert claim_numbers("The top 3 channels were AD1-AD2 and G16-G17.") == []
    assert claim_numbers("AD1-AD2 had 14.2 ripples/min.") == [14.2]


def test_a_fabricated_number_is_struck(verified_store):
    report = ("A-B had 14.2 ripples/min [hfo_001]. A-B had 99.9 ripples/min [hfo_001].")
    result = DeterministicVerifier()(report, verified_store)
    assert result.n_struck == 1 and "99.9" in result.struck[0]["reason"]
    assert "99.9" not in result.kept_text and "14.2" in result.kept_text
    assert result.provenance_coverage == pytest.approx(0.5)


def test_prose_and_caveats_are_never_struck(verified_store):
    report = "Rates come from one short window and physiological ripples occur in healthy tissue."
    result = DeterministicVerifier()(report, verified_store)
    assert result.n_struck == 0 and result.kept_text == report


def test_citing_the_wrong_run_is_caught(verified_store):
    report = "A-B had 14.2 ripples/min [compare_001]."
    result = DeterministicVerifier()(report, verified_store)
    assert result.n_struck == 1 and "did not produce" in result.struck[0]["reason"]


def test_citing_a_run_that_does_not_exist_is_caught(verified_store):
    result = DeterministicVerifier()("A-B had 14.2 ripples/min [hfo_777].", verified_store)
    assert result.n_struck == 1 and "do not exist" in result.struck[0]["reason"]


class _FakeVerifierBackend(Backend):
    """A model verifier that strikes whichever sentence indices it was told to."""

    name = "fake-verifier"

    def __init__(self, strike=(), broken=False):
        self.strike = set(strike)
        self.broken = broken

    def chat(self, messages, tools):
        if self.broken:
            return AssistantMessage(content="I think most of it looks fine, honestly.")
        verdicts = [{"index": i, "supported": i not in self.strike,
                     "run_ids": [], "reason": "cannot resolve"} for i in range(6)]
        return AssistantMessage(content=json.dumps({"verdicts": verdicts}))


def test_the_model_verifier_strikes_what_it_judges_unsupported(verified_store):
    report = "A-B had 14.2 ripples/min [hfo_001]. A-B had 99.9 ripples/min [hfo_001]."
    result = LLMVerifier(_FakeVerifierBackend(strike={1}))(report, verified_store)
    assert result.n_struck == 1 and "99.9" not in result.kept_text


def test_a_broken_verifier_response_keeps_the_report(verified_store):
    """Report quality must not be a function of decoding luck."""
    report = "A-B had 14.2 ripples/min [hfo_001]."
    result = LLMVerifier(_FakeVerifierBackend(broken=True))(report, verified_store)
    assert result.n_struck == 0 and result.kept_text == report


def test_comparing_the_verifiers_measures_what_verification_costs(verified_store):
    # Sentence 0 is true and supported; sentence 1 is fabricated. The model
    # verifier strikes the true one and misses the fabricated one -- both
    # failure modes at once, which is what the comparison is for.
    report = "A-B had 14.2 ripples/min [hfo_001]. A-B had 99.9 ripples/min [hfo_001]."
    delta = compare_verifiers(report, verified_store, _FakeVerifierBackend(strike={0}))
    assert delta["struck_by_llm_only"] == [0]
    assert delta["struck_by_arithmetic_only"] == [1]
    assert delta["coverage_cost"] == 1 and delta["hallucinations_missed_by_llm"] == 1


def test_s3_strikes_an_unsupported_claim_from_the_report(session):
    class Fabricating(ScriptedPlanner):
        name = "fabricating-planner"

        def _report(self, prompt):
            return {"report": "A-B had 12345.6 ripples/min [hfo_001].",
                    "run_ids": ["hfo_001"]}

    session.reset_memo()
    result = run_rung(session, Rung.S3, backend=Fabricating(),
                      stop_rule=FixedBudget(k=4))
    assert result.struck and "12345.6" not in result.report
    assert result.verification["unsupported_claim_rate"] == 1.0


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_a_perfect_ranking_beats_chance():
    labels = SozLabels("x", {"A1", "A2", "B1"}, source="clinical_summary", seizure_free=True)
    channels = ["A1-A2", "B1-B2"] + [f"Z{i}-Z{i + 1}" for i in range(1, 20)]
    score = score_ranking(channels, labels, ks=(2,), n_permutations=500)
    assert score.at_k[2]["precision_at_k"] == 1.0
    assert score.at_k[2]["permutation_p"] < 0.05


def test_an_unlabelled_subject_scores_nothing_rather_than_zero():
    score = score_ranking(["A1-A2"], SozLabels("x"), ks=(1,))
    assert score.at_k == {} and "nothing can be scored" in score.note


def test_the_label_source_is_carried_into_every_score():
    """A number computed against weak labels must never look like a curated one."""
    weak = SozLabels("x", {"A1"}, source="events_markers")
    score = score_ranking(["A1-A2", "B1-B2"], weak, ks=(1,), n_permutations=200)
    assert score.label_source == "events_markers" and score.trustworthy is False


# --------------------------------------------------------------------------
# Stand-in labels, for before the real ones exist
# --------------------------------------------------------------------------


def test_synthetic_labels_recover_the_contacts_events_were_implanted_on(recording):
    """Not an estimate: the simulator knows exactly where it put the ripples."""
    labels = labels_from_ground_truth(recording)
    assert labels.soz_contacts == {c.upper() for c in recording.marked_contacts}
    assert labels.source == "synthetic_truth"


def test_the_ground_truth_rule_is_scale_free_not_an_absolute_floor():
    """An absolute floor sweeps in every background contact that got two events,
    pushing label prevalence to a third of all channels."""
    import pandas as pd

    class _Rec:
        subject = "sim"
        ground_truth = pd.DataFrame({
            "kind": ["ripple"] * 6,
            "contact": ["HOT1"] * 3 + ["HOT2"] * 2 + ["COLD1"]})

    labels = labels_from_ground_truth(_Rec(), min_events=1, min_fraction=0.5)
    assert labels.soz_contacts == {"HOT1", "HOT2"}


def test_artifacts_are_never_a_label(recording):
    """The simulator implants them so the detector can be caught reporting them."""
    ripples = labels_from_ground_truth(recording, kind="ripple").soz_contacts
    artifacts = labels_from_ground_truth(recording, kind="artifact").soz_contacts
    assert ripples and ripples != artifacts


def test_a_real_recording_gets_no_invented_labels():
    """soz_labels never makes anything up; asking for a stand-in is explicit."""
    class _Rec:
        subject = "sub-nobody-999"
        marked_contacts: list = []
        ground_truth = None

    labels = soz_labels("sub-nobody-999", recording=_Rec())
    assert not labels.usable and labels.source == "none"


def test_placeholder_labels_are_reproducible_and_drawn_from_real_channels():
    channels = ["AD1-AD2", "AD2-AD3", "ATT1-ATT2", "G16-G17"]
    first = placeholder_labels("anon-01", channels, n_contacts=3, seed=0)
    again = placeholder_labels("anon-01", channels, n_contacts=3, seed=0)
    assert first.soz_contacts == again.soz_contacts and len(first.soz_contacts) == 3
    assert first.soz_contacts <= {"AD1", "AD2", "AD3", "ATT1", "ATT2", "G16", "G17"}


def test_placeholder_labels_ask_for_more_contacts_than_exist():
    labels = placeholder_labels("anon-01", ["A1-A2"], n_contacts=50)
    assert labels.soz_contacts == {"A1", "A2"}


@pytest.mark.parametrize("source", ["placeholder", "synthetic_truth"])
def test_stand_ins_are_never_trustworthy_and_always_carry_a_warning(source):
    labels = SozLabels("x", {"A1"}, source=source, seizure_free=True)
    assert labels.is_placeholder and not labels.trustworthy and labels.warning


def test_the_warning_travels_into_the_serialised_score():
    """A stand-in reaching a results table unmarked is undetectable later."""
    labels = placeholder_labels("anon-01", ["A1-A2", "B1-B2", "C1-C2"], n_contacts=1)
    payload = score_ranking(["A1-A2", "B1-B2"], labels, ks=(1,),
                            n_permutations=200).as_dict()
    assert payload["is_placeholder"] is True
    assert "PLACEHOLDER" in payload["warning"]
    assert json.dumps(payload)      # survives being written to a results file


def test_the_label_template_round_trips_into_real_labels(tmp_path):
    """Filling the template in and passing it back is the whole swap."""
    channels = ["AD1-AD2", "ATT1-ATT2"]
    stand_in = placeholder_labels("anon-01", channels, n_contacts=2, seed=0)
    path = write_label_template("anon-01", channels, tmp_path / "soz.csv", labels=stand_in)

    text = path.read_text()
    assert "AD1, AD2, ATT1, ATT2" in text      # the exact spelling to use
    assert "PLACEHOLDER" in text               # and that these are not real

    # A centre overwrites soz_contacts and sends it back; the comments stay.
    path.write_text(text.replace(
        f"anon-01,{'; '.join(sorted(stand_in.soz_contacts))}",
        'anon-01,"AD1-2; ATT1"'))
    real = load_labels_csv(path)["anon-01"]
    assert real.soz_contacts == {"AD1", "AD2", "ATT1"}
    assert real.source == "local" and not real.is_placeholder


def test_scoring_synthetic_ground_truth_beats_chance(session):
    """The one case where a non-null score proves the scorer, not the brain."""
    session.reset_memo()
    result = run_rung(session, Rung.S0)
    labels = labels_from_ground_truth(session.recording)
    score = score_ranking(result.top_channels, labels, ks=(5,), n_permutations=1000)
    assert score.at_k[5]["n_hits"] > score.at_k[5]["expected_by_chance"]
    assert score.at_k[5]["permutation_p"] < 0.1


# --------------------------------------------------------------------------
# Falsification: try to make the system confidently wrong
# --------------------------------------------------------------------------


def test_the_null_hypothesis_discriminates_signal_from_noise(recording):
    """A null that always fires is worthless, so check both directions."""
    from onset_hfo.metrics import leader_separation
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.synthetic import make_synthetic_recording

    busy = leader_separation(run_pipeline(recording, verbose=False).rates["rms"])
    empty_recording = make_synthetic_recording(seed=3, duration_s=30, hot_leads=0,
                                               verbose=False)
    empty = leader_separation(run_pipeline(empty_recording, verbose=False).rates["rms"])
    assert busy["distinguishable"] is True, "a recording with implanted ripples has a leader"
    assert empty["distinguishable"] is False, "a recording with nothing in it does not"
    assert empty["n_tied_with_leader"] == empty["n_channels"]
    assert "NO CHANNEL STANDS OUT" in empty["statement"]


def test_leader_separation_says_so_rather_than_guessing_without_intervals():
    from onset_hfo.metrics import leader_separation

    assert leader_separation(None)["available"] is False
    assert leader_separation(pd.DataFrame({"channel": ["A"]}))["available"] is False


def test_the_detect_hfo_tool_carries_the_null(registry):
    run = registry.run("detect_hfo", {"k": 3})
    assert "leader_stands_out" in run.output
    assert run.output["leader_separation"]["available"] is True
    assert "leader_stands_out is false" in run.output["note"]


def test_renaming_channels_changes_nothing_but_the_names(recording):
    from onset_agent.falsify import anonymise_channels
    from onset_hfo.preprocess import prepare

    renamed, mapping = anonymise_channels(recording)
    assert len(mapping) == len(recording.ch_names)
    assert np.allclose(renamed.raw.get_data(), recording.raw.get_data())
    # No clinical meaning survives...
    assert not {n.upper() for n in renamed.raw.ch_names} & {n.upper()
                                                            for n in recording.ch_names}
    # ...but the electrode structure does, or the test compares two montages.
    assert prepare(renamed, verbose=False).n_channels == prepare(
        recording, verbose=False).n_channels


def test_shuffling_permutes_names_without_touching_the_signal(recording):
    from onset_agent.falsify import shuffle_channel_mapping

    shuffled, mapping = shuffle_channel_mapping(recording, seed=1)
    assert sorted(shuffled.raw.ch_names) == sorted(recording.ch_names)
    assert set(mapping.values()) == set(recording.ch_names)
    assert np.allclose(np.sort(shuffled.raw.get_data(), axis=0),
                       np.sort(recording.raw.get_data(), axis=0))


def test_dropping_a_channel_removes_its_contacts(recording):
    from onset_agent.falsify import drop_channel

    reduced = drop_channel(recording, recording.ch_names[0])
    assert len(reduced.raw.ch_names) < len(recording.ch_names)


def test_dropping_a_channel_that_is_not_there_is_an_error(recording):
    from onset_agent.falsify import drop_channel

    with pytest.raises(ValueError, match="not in this recording"):
        drop_channel(recording, "NOSUCH1-NOSUCH2")


def test_the_whole_suite_runs_and_every_test_states_its_expectation(recording):
    from onset_agent.falsify import run_falsification_suite
    from onset_hfo.cohort import soz_labels

    results = run_falsification_suite(recording, soz_labels(recording.subject,
                                                            recording=recording),
                                      repeats=2)
    assert len(results) == 5
    for result in results:
        assert result.expectation, f"{result.name} has no stated expectation"
        assert result.verdict in ("PASS", "FAIL", "INCONCLUSIVE")
        assert result.reading


def test_the_system_survives_every_falsification_attempt(recording):
    """If one of these starts failing, believe the test before the pipeline."""
    from onset_agent.falsify import run_falsification_suite
    from onset_hfo.cohort import soz_labels

    results = run_falsification_suite(recording, soz_labels(recording.subject,
                                                            recording=recording),
                                      repeats=2)
    failed = [r.name for r in results if r.passed is False]
    assert not failed, f"falsification failures: {failed}"


def test_a_recording_with_nothing_in_it_does_not_get_a_leader():
    from onset_agent.falsify import falsify_no_pathology

    result = falsify_no_pathology(duration_s=20)
    assert result.passed is True
    assert result.measure["leader_stands_out"] is False


def test_an_empty_conformal_set_is_a_failure_not_a_narrow_one():
    """A model that rules out every channel is out of distribution, not done.

    Treating zero candidates as "narrow enough to act on" would make the most
    obvious failure mode of a deployed model into its success condition.
    """
    from onset_agent.contract import ToolRun
    from onset_agent.evidence import EvidenceStore
    from onset_agent.planner import ConformalWidth

    rule = ConformalWidth(max_width=3)
    store = EvidenceStore()
    store.append(ToolRun("soz_001", "estimate_soz_probability", {},
                         {"candidate_set_size": 0, "n_channels": 71}))
    stop, reason = rule.should_stop(store, False)
    assert stop, "an empty set should end the run rather than loop"
    assert "EMPTY" in reason and "outside the distribution" in reason
    assert "act on" not in reason.split("not that")[0]


def test_the_real_model_ladder_script_runs_end_to_end(tmp_path, monkeypatch):
    """Exercise scripts/run_model_ladder.py with the scripted planner standing
    in for a served model.

    It cannot be run here with real weights, so this checks the thing that
    would otherwise be discovered forty minutes into someone's GPU run: that
    the script's argument handling, loop, stability pass, falsification pass
    and markdown renderer all work.
    """
    import importlib.util
    import sys
    from pathlib import Path as _Path

    from onset_agent.planner import ScriptedPlanner

    script = _Path(__file__).resolve().parent.parent / "scripts" / "run_model_ladder.py"
    spec = importlib.util.spec_from_file_location("run_model_ladder", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_model_ladder"] = module
    spec.loader.exec_module(module)

    # Stand in for a served model, and skip the reachability probe.
    monkeypatch.setattr("onset_agent.backends.make_backend",
                        lambda *a, **k: ScriptedPlanner())
    monkeypatch.setattr(module, "_check_backend", lambda backend: None)

    out = tmp_path / "ladder"
    code = module.main(["--synthetic", "--duration", "20", "--quick",
                        "--out", str(out)])
    assert code == 0

    payload = json.loads((out / "ladder.json").read_text())
    assert payload["runs"], "no rung produced a result"
    assert all("error" not in run for run in payload["runs"].values()), payload["runs"]

    results = (out / "RESULTS.md").read_text()
    assert "tool calls" in results and "scripted" in results
    assert "—" not in results.split("| configuration |")[1].split("\n")[2], \
        "the table's first data row should carry a real ranking"


# The multiplier ablation, pinned
# --------------------------------------------------------------------------
# EVALUATION.md §6c. The screen takes ~50 minutes, so these read the committed
# extract and the reduction the script applies to it.


@functools.lru_cache(maxsize=1)
def _ablation():
    import sys

    from onset_hfo.config import PROJECT_ROOT

    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_robustness_ablation import analyse

    rows = pd.read_csv(PROJECT_ROOT / "data" / "outcome" / "robustness_ablation.csv")
    return rows, analyse(rows)


def _auc(band, metric, rule, stricter=None):
    _, result = _ablation()
    arm = result[(result.band == band) & (result.metric == metric)
                 & (result.rule == rule)]
    if stricter is not None:
        arm = arm[arm.stricter_x == stricter]
    return arm


def test_the_ablation_covers_both_rules_at_every_retest_point():
    rows, _ = _ablation()
    assert set(rows["band"]) == {"ripple", "fast_ripple"}
    assert set(rows["rule"]) == {"plain", "multiplied"}
    assert rows["subject"].nunique() == 20 and rows["window"].nunique() == 5
    plain = rows[rows.rule == "plain"]
    assert len(plain) == 2 * 20 * 5
    # One multiplied row per re-test point per plain row.
    assert sorted(rows[rows.rule == "multiplied"].stricter_x.unique()) == [1.25, 1.5, 2.0]
    assert len(rows[rows.rule == "multiplied"]) == 3 * len(plain)
    # The survey count is shared, so it must not vary with the rule.
    shared = rows.groupby(["band", "subject", "window"]).n_events_survey.nunique()
    assert (shared == 1).all(), "n_events_survey differs between rules in some window"


def test_the_multiplier_helps_in_the_ripple_band_at_every_retest_point():
    """The positive half of §6c, and the first sign the mechanism carries signal.

    Pinned at every re-test point rather than the best one: a rule that only
    helps at one arbitrary choice of stricter threshold is not a rule.
    """
    for metric in ("top_channel_resected", "tied_set_argmax_resected"):
        plain = float(_auc("ripple", metric, "plain").auc.iloc[0])
        for stricter in (1.25, 1.5, 2.0):
            got = float(_auc("ripple", metric, "multiplied", stricter).auc.iloc[0])
            assert got > plain, (
                f"ripple/{metric} at {stricter}x is now {got:.3f} against plain "
                f"{plain:.3f}; §6c says the multiplier helps here")


def test_the_multiplier_hurts_in_the_fast_ripple_band():
    """The negative half, on the metric that keeps every patient.

    `tied_set_argmax_resected` takes its tied set from the shared survey
    counts, so both rules are scored on the same 13/7 patients and the harm
    cannot be read as survivorship.
    """
    plain = float(_auc("fast_ripple", "tied_set_argmax_resected", "plain").auc.iloc[0])
    arm = _auc("fast_ripple", "tied_set_argmax_resected", "multiplied")
    assert (arm.n_SF == 13).all() and (arm.n_rec == 7).all(), \
        "this metric is only a fair comparator while it keeps every patient"
    assert (arm.auc < plain).all(), \
        f"the fast-ripple degradation is gone: plain {plain:.3f} vs {list(arm.auc)}"


def test_a_silent_stricter_pass_annihilates_the_ranking():
    """The defect §6c asks to fix, pinned so a fix has a failing test to flip.

    When a sparse band finds nothing at the stricter threshold, robustness is 0
    on every channel and the score is uniformly zero -- leaving no leader at
    all. A robustness of 0 from "no events here" means *unmeasured*, not
    *refuted*, and the rule does not currently make that distinction.
    """
    rows, _ = _ablation()
    mult = rows[rows.rule == "multiplied"]
    collapsed = mult[mult.score_mass == 0]

    assert len(collapsed) > 0, \
        "if the collapse is gone the rule was fixed; update §6c and this test"
    # Overwhelmingly a sparse-band failure.
    assert (collapsed.band == "fast_ripple").mean() > 0.9
    # And the collapse is exactly what produces a missing answer.
    assert collapsed.top_channel_resected.isna().all()
    assert mult[mult.score_mass > 0].top_channel_resected.notna().all(), \
        "a window with score left has no excuse for a missing leader"


def test_the_ablations_copy_of_the_rule_still_matches_the_planners():
    """The ablation transcribes `rank_channels` rather than calling it, because
    the planner reads an evidence store and the ablation reads count series.
    Two copies of a rule is how one of them gets fixed and the other does not,
    so they are pinned against each other -- on the ordinary case, and on the
    silent re-test that is the whole reason the rule was changed.
    """
    import sys

    from onset_hfo.config import PROJECT_ROOT

    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_robustness_ablation import _robustness

    survey = pd.Series({"Z1-Z2": 12.0, "M1-M2": 6.0, "A1-A2": 3.0})

    def planner_factors(stricter: dict, n_accepted: int) -> dict:
        store = EvidenceStore(subject="sub-01")
        store.append(ToolRun(run_id="s", tool="detect_hfo", input={}, output={
            "threshold_sd": 3.0, "n_accepted": int(survey.sum()),
            "channels": {c: {"rate_per_min": v} for c, v in survey.items()}}))
        store.append(ToolRun(run_id="x", tool="detect_hfo", input={}, output={
            "threshold_sd": 5.0, "n_accepted": n_accepted,
            "channels": {c: {"rate_per_min": v} for c, v in stricter.items()}}))
        return {r.channel: r.robustness for r in rank_channels(store)}

    # A real re-test: both demote the same channels by the same factors.
    partial = {"Z1-Z2": 9.0, "M1-M2": 0.0, "A1-A2": 0.0}
    factors, silent = _robustness(survey, pd.Series(partial))
    assert silent is False
    assert factors == pytest.approx(planner_factors(partial, 9))

    # A silent re-test: both refuse to fire, so no score is annihilated.
    quiet = dict.fromkeys(survey.index, 0.0)
    factors, silent = _robustness(survey, pd.Series(quiet))
    assert silent is True
    assert factors == pytest.approx(planner_factors(quiet, 0))
    assert set(factors.values()) == {1.0}


def test_the_tie_aware_metric_is_undefined_for_a_multiplied_score():
    """Structural, and independent of every AUC in §6c.

    `candidate_channels` decides ties by overlapping Poisson intervals, which
    needs an integer count. `rate x robustness` is not one, so the project's own
    tie-aware metric cannot be computed for the planner's own score -- and is
    left NaN rather than computed on a truncated count.
    """
    rows, _ = _ablation()
    assert rows[rows.rule == "plain"].candidates_resected.notna().any()
    assert rows[rows.rule == "multiplied"].candidates_resected.isna().all(), \
        "candidates_resected must stay NaN for the multiplied rule"


def test_no_arm_of_the_ablation_clears_its_power_floor():
    """So the ripple gain is 'consistent with', not 'demonstrated'.

    Floors: 0.85 at 13/7, 0.87 at 11/7 and 13/6, 0.88 at 9/6 -- recorded rather
    than recomputed, as in `test_populations.py`, because the simulation is slow.
    """
    floors = {(13, 7): 0.85, (11, 7): 0.87, (13, 6): 0.87, (9, 6): 0.88}
    _, result = _ablation()
    sizes = {(int(r.n_SF), int(r.n_rec)) for r in result.itertuples()}
    assert sizes <= set(floors), f"unrecorded group size in {sizes}"

    lowest = min(floors[s] for s in sizes)
    worst = result.loc[result.auc.idxmax()]
    assert worst.auc < lowest, (
        f"{worst.band}/{worst.metric}/{worst.rule} now reaches {worst.auc:.3f} "
        f"against a floor of at least {lowest}; §6c's hedging needs revisiting")
