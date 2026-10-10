"""Command line interface.

    python -m onset_hfo.cli run --synthetic              # offline, labelled data
    python -m onset_hfo.cli run                          # the public example slice
    python -m onset_hfo.cli run --subject sub-pt01 --task ictal --run 01 \
                               --start 50 --stop 110 --figures
    python -m onset_hfo.cli evaluate --seeds 1 7 42      # measure the detectors
    python -m onset_hfo.cli runs --subject sub-pt01      # what else is in the archive
    python -m onset_hfo.cli fetch --subject sub-01       # cache a window for the reviewer

Every command prints where it wrote its results, because the agent
(``python -m onset_agent.cli --results <dir>``) reads exactly that directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from onset_hfo.benchmark import DEFAULT_THRESHOLDS
from onset_hfo.config import (
    DATASETS,
    DEFAULT_RUN,
    DEFAULT_SUBJECT,
    DEFAULT_TASK,
    DEFAULT_TSTART,
    DEFAULT_TSTOP,
    PIPELINE_VERSION,
    RESULTS_DIR,
    THRESHOLDS,
    PipelineConfig,
    ensure_dirs,
)
from onset_hfo.detectors.base import Event
from onset_hfo.outcome import FULL_RUN_S
from onset_hfo.stability import DISJOINT_LENGTH, GROWING_WINDOWS


def _cmd_run(args: argparse.Namespace) -> int:
    from onset_hfo.pipeline import run_pipeline

    ensure_dirs()
    if args.synthetic:
        from onset_hfo.synthetic import make_synthetic_recording
        recording = make_synthetic_recording(seed=args.seed, duration_s=args.duration)
    else:
        from onset_hfo.datasets import fetch_slice
        recording = fetch_slice(subject=args.subject, task=args.task, run=args.run,
                                t_start=args.start, t_stop=args.stop)

    cfg = PipelineConfig()
    cfg.top_k = args.top_k
    if args.threshold:
        value = THRESHOLDS.get(args.threshold)
        if value is None:
            try:
                value = float(args.threshold)
            except ValueError:
                print(f"[onset-hfo] --threshold must be a number or one of: "
                      f"{', '.join(THRESHOLDS)}")
                return 2
        cfg.rms.threshold_sd = value
        cfg.line_length.threshold_sd = value
        print(f"[onset-hfo] detection threshold {value:g} SD ({args.threshold})")
    result = run_pipeline(recording, cfg, with_spikes=not args.no_spikes,
                          save_to=args.out or RESULTS_DIR)
    out_dir = Path(args.out or RESULTS_DIR) / \
        f"{recording.subject}_{recording.task}_run-{recording.run}"

    if args.figures:
        from onset_hfo.viz import save_all_figures
        written = save_all_figures(result, out_dir / "figures")
        print(f"[onset-hfo] {len(written)} figures in {out_dir / 'figures'}")

    if recording.ground_truth is not None:
        from onset_hfo.evaluate import evaluate_detections
        print("\n[onset-hfo] synthetic ground truth is available, so scores can be computed:")
        for name, events in result.events.items():
            print("   " + evaluate_detections(events, recording.ground_truth, detector=name).summary())
        if result.spikes:
            print("   " + evaluate_detections(result.spikes, recording.ground_truth,
                                              detector="spike", kind="spike").summary())

    print("\n" + result.report.to_markdown().split("## Findings")[0])
    print(f"[onset-hfo] full report: {out_dir / 'report.md'}")
    print(f"[onset-hfo] ask the agent about it:  "
          f"python -m onset_agent.cli --results {out_dir} "
          f"--question 'which channels have the highest ripple rate?'")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Measure the detectors against synthetic ground truth, over several seeds."""
    import pandas as pd

    from onset_hfo.evaluate import evaluate_detections, validation_benefit
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.synthetic import make_synthetic_recording

    rows = []
    for seed in args.seeds:
        rec = make_synthetic_recording(seed=seed, duration_s=args.duration, verbose=False)
        result = run_pipeline(rec, verbose=False)
        for name, events in result.events.items():
            res = evaluate_detections(events, rec.ground_truth, detector=name)
            rows.append({"seed": seed, **res.as_dict()})
        if result.spikes:
            res = evaluate_detections(result.spikes, rec.ground_truth, detector="spike", kind="spike")
            rows.append({"seed": seed, **res.as_dict()})
        if args.verbose:
            print(f"seed {seed}:")
            print(validation_benefit(result.events["rms"], rec.ground_truth).to_string(index=False))
    table = pd.DataFrame(rows)
    summary = table.groupby(["detector", "kind"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    print("\n[onset-hfo] scores over seeds " + ", ".join(str(s) for s in args.seeds))
    print(summary.round(3).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"[onset-hfo] per-seed scores written to {args.out}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Score the detectors against expert HFO markings on a real cohort."""
    from onset_hfo.benchmark import benchmark_cohort

    ensure_dirs()
    result = benchmark_cohort(
        subjects=args.subjects, n_subjects=args.n_subjects, dataset=args.dataset,
        run=args.run, t_start=args.start, t_stop=args.stop,
        thresholds=tuple(args.thresholds), detectors=tuple(args.detectors),
        bands=tuple(args.bands))
    if not len(result.scores):
        print("[onset-hfo] nothing scored: no subject produced expert markings")
        return 1
    for band in args.bands:
        table = result.summary(band)
        if not len(table):
            continue
        print(f"\n[onset-hfo] {band} band, cohort means over "
              f"{result.subjects['subject'].nunique()} subjects "
              f"({args.stop - args.start:g} s each):")
        print(table.to_string(index=False))
    print("\n[onset-hfo] operating point the data prefers:")
    print(f"   by F1:              {result.best_threshold(args.bands[0], 'f1')}")
    print(f"   by rank agreement:  {result.best_threshold(args.bands[0], 'rank')}")
    result.save(args.out)
    return 0


def _cmd_outcome(args: argparse.Namespace) -> int:
    """Ask whether the HFO map points at the tissue whose removal cured the patient."""
    from onset_hfo.outcome import outcome_study

    ensure_dirs()
    threshold = args.threshold if args.threshold is not None else None
    result = outcome_study(
        subjects=args.subjects, n_subjects=args.n_subjects, dataset=args.dataset,
        run=args.run, t_start=args.start, t_stop=args.stop,
        detector=args.detector, threshold_sd=threshold, bands=tuple(args.bands),
        drop_eloquent=not args.keep_eloquent)
    if not len(result.subjects):
        print("[onset-hfo] nothing measured: no subject had both a resected zone and a recording")
        return 1
    for metric in ("share_in_rz", "top_channel_resected", "candidates_resected",
                   "top3_resected"):
        print(f"\n[onset-hfo] {metric}, seizure-free vs recurrence:")
        print(result.summary(metric).to_string(index=False))
    decided = result.subjects.query("scope == 'reviewed'")
    if "leader_alone" in decided.columns and len(decided):
        print("\n[onset-hfo] how often did the data actually pick ONE channel?")
        table = (decided.groupby(["source", "band"])
                 .agg(patients=("leader_alone", "count"),
                      leader_alone=("leader_alone", "sum"),
                      median_candidates=("n_candidates", "median"),
                      max_candidates=("n_candidates", "max")))
        print(table.to_string())
    metric, band = result.PRIMARY
    if band in args.bands:
        print(f"\n[onset-hfo] pre-specified comparison ({metric}, {band} band):")
        print(f"   {result.verdict()}")
    for other_band in args.bands:
        for other_metric in ("share_in_rz", "top_channel_resected"):
            if (other_metric, other_band) == result.PRIMARY:
                continue
            print(f"\n[onset-hfo] {other_metric}, {other_band} band: "
                  f"{result.verdict(metric=other_metric, band=other_band)}")
    result.save(args.out)
    return 0


def _cmd_stability(args: argparse.Namespace) -> int:
    """Ask whether the channel ranking is stable enough to carry the claim."""
    from onset_hfo.stability import across_runs, plot_stability, stability_study

    ensure_dirs()
    if args.across_runs:
        result = across_runs(runs_per_subject=args.across_runs, dataset=args.dataset,
                             detector=args.detector, bands=tuple(args.bands),
                             subjects=args.subjects, prune_cache=not args.keep_cache)
        if not len(result.groups):
            print("[onset-hfo] nothing measured")
            return 1
        print("\n[onset-hfo] AUC spread across runs (nights):")
        print(result.spread(arm="run").to_string(index=False))
        print("\n[onset-hfo] does the per-patient answer hold from run to run?")
        print(result.decision_stability(arm="run").to_string(index=False))
        pooled = result.groups.query("arm == 'pooled' and scope == 'reviewed'")
        if len(pooled):
            print("\n[onset-hfo] runs pooled per patient:")
            print(pooled[["source", "band", "metric", "n_seizure_free", "n_recurrence",
                          "mean_seizure_free", "mean_recurrence", "auc", "auc_lo",
                          "auc_hi", "p_permutation"]].round(3).to_string(index=False))
        print(f"\n[onset-hfo] {result.verdict_runs()}")
        result.save(args.out)
        return 0
    result = stability_study(
        growing=tuple(args.growing), disjoint_length=args.disjoint,
        dataset=args.dataset, detector=args.detector, bands=tuple(args.bands),
        subjects=args.subjects)
    if not len(result.groups):
        print("[onset-hfo] nothing measured")
        return 1
    print("\n[onset-hfo] growing windows, "
          f"{result.primary[0]} / {result.primary[1]}:")
    print(result.curve().to_string(index=False))
    print(f"\n[onset-hfo] spread across disjoint {args.disjoint:g} s windows:")
    print(result.spread().to_string(index=False))
    stability = result.top_channel_stability()
    if len(stability):
        print("\n[onset-hfo] is the busiest channel the same channel?")
        print(stability.to_string(index=False))
    out = result.save(args.out)
    if not args.no_figure:
        plot_stability(result, out / "window_stability.png")
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    """Detect over a recording too long to hold in memory.

    Writes events and per-channel rates. It deliberately does **not** write a
    report or figures: both need the whole signal array, which is the thing
    this command exists because you do not have. Use ``run`` on a window you
    can hold when you want the report.
    """
    import json

    from onset_hfo.datasets import iter_slices
    from onset_hfo.detectors.base import events_to_frame
    from onset_hfo.metrics import channel_rates
    from onset_hfo.preprocess import prepare
    from onset_hfo.streaming import OverlapTooShort, stream_detect

    ensure_dirs()
    cfg = PipelineConfig()
    cfg.top_k = args.top_k

    seen: dict[str, object] = {}

    def chunks():
        for plan, recording in iter_slices(
                t_start=args.start, t_stop=args.stop, chunk_s=args.chunk,
                overlap_s=args.overlap, subject=args.subject, task=args.task,
                run=args.run, verbose=args.verbose):
            prep = prepare(recording, cfg.preprocess, verbose=False)
            seen.setdefault("channels", list(prep.ch_names))
            seen.setdefault("steps", list(prep.steps))
            seen.setdefault("recording", recording)
            yield plan, prep

    try:
        events = stream_detect(chunks, cfg, detectors=tuple(args.detectors))
    except OverlapTooShort as exc:
        print(f"[onset-hfo] {exc}")
        return 2

    duration = args.stop - args.start
    recording = seen.get("recording")
    out_dir = Path(args.out or RESULTS_DIR) / (
        f"{args.subject}_stream_{args.start:g}-{args.stop:g}s")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [events_to_frame(evs) for evs in events.values() if evs]
    if frames:
        import pandas as pd
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "events.csv", index=False)
    for name, evs in events.items():
        channel_rates(evs, duration, seen.get("channels")).to_csv(
            out_dir / f"rates_{name}.csv", index=False)

    provenance = recording.provenance() if recording is not None else {}
    provenance.update({
        "analysis": "streamed",
        "t_start_s": args.start, "t_stop_s": args.stop,
        "chunk_s": args.chunk, "overlap_s": args.overlap,
        "preprocessing": seen.get("steps", []),
        "note": ("Baselines were measured once over the whole window, not per "
                 "chunk, so these events do not depend on the chunk size. See "
                 "onset_hfo/streaming.py."),
    })
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    print(f"\n[onset-hfo] {duration / 60:.1f} minutes analysed in "
          f"{args.chunk:g} s chunks")
    for name, evs in events.items():
        print(f"[onset-hfo]   {name}: {len(evs)} events "
              f"({60 * len(evs) / duration:.1f}/min across all channels)")
    print(f"[onset-hfo] written to {out_dir}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """The headless half of the hand-annotated benchmark (roadmap item 5).

    Sampling, agreement and scoring run here; the marking itself is
    ``notebooks/07_annotation.ipynb``, because looking at a waveform is not a
    thing a command line does well.
    """
    import json

    import pandas as pd

    from onset_hfo.review import (
        agreement,
        ceiling_note,
        read_manifest,
        sample_windows,
        save_study,
        score_against_annotations,
        write_manifest,
    )
    from onset_hfo.store import ResultStore

    out = Path(args.out)
    if args.step == "sample":
        store = ResultStore(args.analysis)
        events = store.events
        by_detector = {}
        for name in sorted(events.detector.unique()):
            if name == "spike":
                continue
            rows = events[events.detector == name]
            by_detector[name] = [Event(
                channel=r.channel, start=r.start, stop=r.stop, detector=name,
                band=(r.band_low, r.band_high), accepted=bool(r.accepted))
                for r in rows.itertuples()]
        start = float(store.provenance.get("slice_start_s", 0.0))
        duration = float(store.provenance.get("slice_stop_s", 60.0)) - start
        windows = sample_windows(by_detector, duration_s=duration,
                                 channels=sorted(events.channel.unique()),
                                 n_per_stratum=args.per_stratum, seed=args.seed,
                                 t_offset=start)
        path = write_manifest(windows, out / "manifest.csv")
        counts = pd.Series([w.stratum for w in windows]).value_counts()
        print(f"[onset-hfo] {len(windows)} windows: "
              + ", ".join(f"{k} {v}" for k, v in counts.items()))
        print(f"[onset-hfo] give the reviewer   {path}")
        print(f"[onset-hfo] keep back the key   {path.with_suffix('.key.csv')}")
        return 0

    annotations = pd.read_csv(out / "annotations.csv")
    stats = agreement(annotations)
    print("\n" + ceiling_note(stats) + "\n")
    if args.step == "agreement":
        print(json.dumps(stats, indent=2))
        return 0

    manifest = read_manifest(out / "manifest.csv")
    key = pd.read_csv(out / "manifest.key.csv")
    store = ResultStore(args.analysis)
    rows = store.events[store.events.detector == args.detector]
    events = [Event(channel=r.channel, start=r.start, stop=r.stop, detector=args.detector,
                    band=(r.band_low, r.band_high), accepted=bool(r.accepted))
              for r in rows.itertuples()]
    scores = {}
    for rule in ("both", "either"):
        scores[rule] = score_against_annotations(events, manifest, annotations, key,
                                                 consensus=rule, detector=args.detector)
        print(f"consensus = {rule}")
        for stratum, s in scores[rule]["strata"].items():
            print(f"  {stratum:11s} n={s['n']:3d}  tp={s['true_positive']:3d} "
                  f"fp={s['false_positive']:3d} fn={s['false_negative']:3d} "
                  f"tn={s['true_negative']:3d}  reviewers called "
                  f"{s['reviewer_hfo_rate']:.0%} of them HFOs")
    save_study(out, stats, scores)
    print(f"\n[onset-hfo] written to {out}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Put one window in the cache, so the reviewer can open it offline.

    This is the only command here whose point is the *side effect*: it analyses
    nothing and prints nothing but a path. The desktop reviewer refuses to
    reach for the network on its own -- a clinician should never discover
    mid-click that the thing they chose needs 700 MB over a hospital
    connection -- so something has to fill the cache first, and this is it.

    Its dataset default differs from every other command's, deliberately.
    `DATASET` is `ds003029`, the ictal archive, whose subjects are named
    `sub-pt01`; the reviewer is built around `ds003498`, whose subjects are
    `sub-01` and which is the one carrying expert HFO markings. A `fetch`
    that inherited the global default would send `--subject sub-01` to the
    wrong archive and fail with a 404 about a path nobody asked for.
    """
    from onset_hfo.datasets import fetch_slice

    ensure_dirs()
    try:
        recording = fetch_slice(
            dataset=args.dataset, subject=args.subject, task=args.task,
            run=args.run, t_start=args.t_start, t_stop=args.t_stop,
            force=args.force, verbose=True)
    except Exception as problem:
        print(f"[onset-hfo] could not fetch {args.subject} "
              f"{args.t_start:g}-{args.t_stop:g} s from {args.dataset}: "
              f"{type(problem).__name__}: {problem}", file=sys.stderr)
        return 1

    print(f"[onset-hfo] cached {recording.subject} "
          f"{args.t_start:g}-{args.t_stop:g} s: {len(recording.ch_names)} "
          f"channels at {recording.sfreq:g} Hz, {recording.duration:g} s")
    print(f"[onset-hfo] open it with:  onset-review --subject {recording.subject} "
          f"--window {args.t_start:g} {args.t_stop:g}")
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    from onset_hfo.datasets import list_runs

    table = list_runs(args.subject)
    print(table.to_string(index=False) if len(table) else "no runs found")
    return 0


def _cmd_openneuro(args: argparse.Namespace) -> int:
    """List, describe or fetch any OpenNeuro dataset (`onset_hfo.openneuro`)."""
    import pandas as pd

    from onset_hfo import openneuro

    try:
        if args.action == "catalogue":
            if args.refresh:
                openneuro.build_catalogue(out=None)
            table = openneuro.catalogue(modality=args.kind, search=args.search,
                                        data_home=args.data_home)
            with pd.option_context("display.max_rows", None, "display.width", 200,
                                   "display.max_colwidth", 80):
                print(table[["dataset_id", "modalities", "subjects", "name"]]
                      .to_string(index=False) if len(table) else "no dataset matches")
            return 0
        if not args.dataset:
            raise ValueError(f"'{args.action}' needs a dataset id, such as ds004100; "
                             "'catalogue --search …' finds one")
        if args.action == "describe":
            print(openneuro.describe_openneuro(args.dataset, data_home=args.data_home).DESCR)
        elif args.action == "list":
            table = openneuro.list_openneuro(args.dataset, subject=args.subject,
                                             data_home=args.data_home)
            with pd.option_context("display.max_rows", None, "display.width", 200):
                print(table.drop(columns="path").to_string(index=False) if len(table)
                      else "no continuous recordings found")
        else:
            def show(fraction: float, message: str) -> None:
                # One line, rewritten in place, on stderr: stdout stays the path.
                print(f"\r[onset-hfo]   {message} ({100 * fraction:.0f}%)",
                      end="\n" if fraction >= 1.0 else "", file=sys.stderr, flush=True)

            bunch = openneuro.fetch_openneuro(
                args.dataset, args.subject, session=args.session, task=args.task,
                acq=args.acq, run=args.run, t_start=args.t_start,
                t_stop=None if args.t_stop < 0 else args.t_stop,
                data_home=args.data_home, max_mb=args.max_mb, progress=show)
            print(bunch.local_path)
    except (ValueError, OSError, RuntimeError) as problem:
        print(f"[onset-hfo] {problem}", file=sys.stderr)
        return 1
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from onset_hfo.store import ResultStore

    store = ResultStore(args.results)
    print(json.dumps(store.metadata(), indent=2))
    print((Path(args.results) / "report.md").read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="onset-hfo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"onset-hfo {PIPELINE_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the detection pipeline on one recording")
    run.add_argument("--synthetic", action="store_true",
                     help="use the labelled synthetic recording instead of the public archive")
    run.add_argument("--subject", default=DEFAULT_SUBJECT)
    run.add_argument("--task", default=DEFAULT_TASK)
    run.add_argument("--run", default=DEFAULT_RUN)
    run.add_argument("--start", type=float, default=DEFAULT_TSTART,
                     help="start of the slice, seconds into the recording")
    run.add_argument("--stop", type=float, default=DEFAULT_TSTOP)
    run.add_argument("--duration", type=float, default=120.0,
                     help="length of the synthetic recording, seconds")
    run.add_argument("--seed", type=int, default=7, help="synthetic recording seed")
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--threshold", default=None, metavar="SD|PRESET",
                     help="detection threshold in robust SDs, or a measured preset: "
                          + ", ".join(f"{k} ({v:g})" for k, v in THRESHOLDS.items()))
    run.add_argument("--no-spikes", action="store_true", help="skip the discharge detector")
    run.add_argument("--figures", action="store_true", help="also render the standard figures")
    run.add_argument("--out", default=None, help=f"output directory (default: {RESULTS_DIR})")
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("evaluate", help="score the detectors against synthetic ground truth")
    ev.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    ev.add_argument("--duration", type=float, default=120.0)
    ev.add_argument("--out", default=None, help="write per-seed scores to this CSV")
    ev.add_argument("--verbose", action="store_true")
    ev.set_defaults(func=_cmd_evaluate)

    bench = sub.add_parser(
        "benchmark",
        help="score the detectors against expert HFO markings (ds003498)")
    bench.add_argument("--dataset", default="ds003498",
                       help="dataset with expert markings (default: ds003498)")
    bench.add_argument("--subjects", nargs="+", default=None,
                       help="subject labels; default: every subject in the dataset")
    bench.add_argument("--n-subjects", type=int, default=None,
                       help="use only the first N subjects (a quick look)")
    bench.add_argument("--run", default="01")
    bench.add_argument("--start", type=float, default=0.0)
    bench.add_argument("--stop", type=float, default=60.0,
                       help="seconds of each recording to score (default 60)")
    bench.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                       help="detection thresholds to sweep, in robust SDs")
    bench.add_argument("--detectors", nargs="+", default=["rms", "line_length"])
    bench.add_argument("--bands", nargs="+", default=["ripple"],
                       choices=["ripple", "fast_ripple"])
    bench.add_argument("--out", default=None, help=f"output directory (default: {RESULTS_DIR})")
    bench.set_defaults(func=_cmd_benchmark)

    review = sub.add_parser(
        "review",
        help="sample, agree and score a hand-annotated benchmark (marking is "
             "notebooks/07_annotation.ipynb)")
    review.add_argument("step", choices=["sample", "agreement", "score"])
    review.add_argument("--analysis", default=None,
                        help="a saved analysis directory (needed by sample and score)")
    review.add_argument("--out", default="artifacts/review")
    review.add_argument("--per-stratum", type=int, default=70,
                        help="windows per stratum; three strata, so 70 means 210")
    review.add_argument("--detector", default="rms")
    review.add_argument("--seed", type=int, default=0)
    review.set_defaults(func=_cmd_review)

    stream = sub.add_parser(
        "stream",
        help="detect over a recording longer than memory, in overlapping chunks")
    stream.add_argument("--subject", default=None)
    stream.add_argument("--task", default=None)
    stream.add_argument("--run", default=None)
    stream.add_argument("--start", type=float, default=0.0)
    stream.add_argument("--stop", type=float, required=True,
                        help="seconds; the whole window to analyse")
    stream.add_argument("--chunk", type=float, default=60.0,
                        help="seconds per chunk (default: 60)")
    stream.add_argument("--overlap", type=float, default=2.0,
                        help="seconds read beyond each chunk and discarded, so a "
                             "boundary cannot truncate an event or a filter "
                             "(default: 2)")
    stream.add_argument("--detectors", nargs="+", default=["rms", "line_length"],
                        choices=["rms", "line_length", "hilbert", "short_time_energy"])
    stream.add_argument("--top-k", type=int, default=5)
    stream.add_argument("--out", default=None)
    stream.add_argument("--verbose", action="store_true",
                        help="print each chunk as it is fetched")
    stream.set_defaults(func=_cmd_stream)

    out = sub.add_parser(
        "outcome",
        help="test the HFO map against post-surgical seizure outcome (ds003498)")
    out.add_argument("--dataset", default="ds003498",
                     help="dataset with a resected zone and outcomes (default: ds003498)")
    out.add_argument("--subjects", nargs="+", default=None)
    out.add_argument("--n-subjects", type=int, default=None)
    out.add_argument("--run", default="01")
    out.add_argument("--start", type=float, default=0.0)
    out.add_argument("--stop", type=float, default=FULL_RUN_S,
                     help="seconds of each recording to use. The default is the whole "
                          "run; 60 gives a different answer, which is itself a result "
                          "(see docs/OUTCOME.md)")
    out.add_argument("--detector", default="rms", choices=["rms", "line_length"])
    out.add_argument("--threshold", type=float, default=None,
                     help="one threshold for every band; default is the measured "
                          "per-band operating point (2.0 SD ripples, 5.0 SD fast ripples)")
    out.add_argument("--bands", nargs="+", default=["ripple", "fast_ripple"],
                     choices=["ripple", "fast_ripple"])
    out.add_argument("--keep-eloquent", action="store_true",
                     help="keep contacts the source study excluded for evoked "
                          "motor or language responses (default: drop them)")
    out.add_argument("--out", default=None, help=f"output directory (default: {RESULTS_DIR})")
    out.set_defaults(func=_cmd_outcome)

    stab = sub.add_parser(
        "stability",
        help="is the channel ranking stable across analysis windows? (ds003498)")
    stab.add_argument("--dataset", default="ds003498")
    stab.add_argument("--subjects", nargs="+", default=None)
    stab.add_argument("--growing", type=float, nargs="+", default=list(GROWING_WINDOWS),
                      help="window ends for the growing arm, in seconds")
    stab.add_argument("--disjoint", type=float, default=DISJOINT_LENGTH,
                      help="length of each equal, non-overlapping window")
    stab.add_argument("--detector", default="rms", choices=["rms", "line_length"])
    stab.add_argument("--bands", nargs="+", default=["ripple", "fast_ripple"],
                      choices=["ripple", "fast_ripple"])
    stab.add_argument("--across-runs", type=int, default=0, metavar="N",
                      help="instead of windows, score up to N runs per subject and "
                           "pool them: does the answer hold from one night to the "
                           "next? ds003498 has 385 runs (~46 GB), so slices are "
                           "deleted after use unless --keep-cache")
    stab.add_argument("--keep-cache", action="store_true",
                      help="with --across-runs, keep each downloaded slice")
    stab.add_argument("--no-figure", action="store_true")
    stab.add_argument("--out", default=None, help=f"output directory (default: {RESULTS_DIR})")
    stab.set_defaults(func=_cmd_stability)

    fetch = sub.add_parser(
        "fetch", help="download one window into the cache, for the reviewer to "
                      "open offline")
    # ds003498 rather than the global default: see _cmd_fetch.
    fetch.add_argument("--dataset", default="ds003498", choices=sorted(DATASETS),
                       help="(default: %(default)s, the archive with expert "
                            "HFO markings)")
    fetch.add_argument("--subject", default="sub-01")
    fetch.add_argument("--task", default=None)
    fetch.add_argument("--run", default=None)
    fetch.add_argument("--t-start", type=float, default=0.0,
                       help="seconds into the recording (default: %(default)s)")
    fetch.add_argument("--t-stop", type=float, default=60.0)
    fetch.add_argument("--force", action="store_true",
                       help="download again even if this window is cached")
    fetch.set_defaults(func=_cmd_fetch)

    runs = sub.add_parser("runs", help="list the runs available for a subject in the archive")
    runs.add_argument("--subject", default=DEFAULT_SUBJECT)
    runs.set_defaults(func=_cmd_runs)

    neuro = sub.add_parser(
        "openneuro", help="list, describe or fetch any OpenNeuro dataset's recordings")
    neuro.add_argument("action", choices=("catalogue", "list", "describe", "fetch"),
                       help="catalogue: find datasets by name; the others take a dataset id")
    neuro.add_argument("dataset", nargs="?", default=None,
                       help="an OpenNeuro id, such as ds004100")
    neuro.add_argument("--search", default=None,
                       help="catalogue: words every dataset's id or name must contain")
    neuro.add_argument("--kind", choices=("ieeg", "eeg"), default=None,
                       help="catalogue: only datasets with this kind of recording")
    neuro.add_argument("--refresh", action="store_true",
                       help="catalogue: survey OpenNeuro again (a few minutes) rather than "
                            "read the copy bundled with the package")
    neuro.add_argument("--subject", default=None)
    neuro.add_argument("--session", default=None)
    neuro.add_argument("--task", default=None)
    neuro.add_argument("--acq", default=None)
    neuro.add_argument("--run", default=None)
    neuro.add_argument("--t-start", type=float, default=0.0,
                       help="seconds into the recording (default: %(default)s)")
    neuro.add_argument("--t-stop", type=float, default=60.0,
                       help="seconds into the recording, or -1 for its end "
                            "(default: %(default)s)")
    neuro.add_argument("--max-mb", type=float, default=500.0,
                       help="refuse a download larger than this (default: %(default)s)")
    neuro.add_argument("--data-home", default=None,
                       help="where downloads are kept (default: $ONSET_HFO_DATA or the "
                            "package's data cache)")
    neuro.set_defaults(func=_cmd_openneuro)

    rep = sub.add_parser("report", help="print a saved report")
    rep.add_argument("results", help="a results directory written by 'run'")
    rep.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
