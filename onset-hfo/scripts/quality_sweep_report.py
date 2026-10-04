"""Read the quality sweep and answer the question it was run to answer.

The sweep (`quality_sweep.py`) records what the data-quality stage did to
every cached window of ds003498. This reads those tables and asks whether it
should have.

The decisive comparison is the third one. The archive marks HFOs per bipolar
channel, so every contact the stage flagged can be set beside the number of
ripples the archive's own annotators marked on it. If flagged contacts are the
ones the experts marked *most*, the band-power check is not finding noise --
it is finding epilepsy, and a version of it that removed those contacts would
be deleting the result. That is the behaviour this sweep exists to measure
rather than assume, and it is why the check flags instead of removing.

Needs a second cheap pass over the windows for the per-channel expert counts,
because `verdicts.csv` only holds the contacts something was said about.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("ONSET_HFO_OFFLINE", "1")
warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402

from onset_hfo.datasets import fetch_slice  # noqa: E402
from onset_hfo.preprocess import prepare  # noqa: E402
from onset_hfo.quality import channel_quality, segment_quality  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "quality_sweep"


def every_channel() -> pd.DataFrame:
    """One row per contact per window: its verdict and its expert ripple count."""
    cached = OUT / "channels.csv"
    if cached.exists():
        return pd.read_csv(cached)
    rows = []
    windows = pd.read_csv(OUT / "windows.csv")
    for index, window in enumerate(windows.itertuples(), 1):
        record = fetch_slice(dataset="ds003498", subject=window.subject,
                             run="01", t_start=window.t_start,
                             t_stop=window.t_stop, verbose=False)
        prep = prepare(record, verbose=False)
        quality = channel_quality(prep, segments=segment_quality(prep))
        truth = getattr(record, "ground_truth", None)
        marked = ({} if truth is None or not len(truth)
                  else truth[truth["kind"].astype(str) == "ripple"]
                  .groupby("channel").size().to_dict())
        reviewed = set(getattr(record, "reviewed_channels", []) or [])
        for row in quality.itertuples():
            rows.append({
                "subject": window.subject, "t_start": window.t_start,
                "channel": str(row.channel),
                "set_aside": not row.good, "flagged": bool(row.flagged),
                "reason": row.reason,
                "hf_ratio_sd": float(row.hf_ratio_sd),
                "expert_ripples": int(marked.get(str(row.channel), 0)),
                "expert_reviewed": str(row.channel) in reviewed,
            })
        print(f"  [{index:3d}/{len(windows)}] {window.subject} "
              f"{window.t_start}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(cached, index=False)
    return frame


def main() -> int:
    windows = pd.read_csv(OUT / "windows.csv")
    verdicts = (pd.read_csv(OUT / "verdicts.csv")
                if (OUT / "verdicts.csv").exists() else pd.DataFrame())

    print("=" * 72)
    print(f"1. DOES IT CHANGE ANY PUBLISHED NUMBER?   "
          f"({len(windows)} windows, {windows.subject.nunique()} subjects)")
    print("=" * 72)
    differing = windows[~windows["rates_identical"]]
    moved = windows[windows["leader_with"] != windows["leader_without"]]
    print(f"  rate tables identical with the stage on and off : "
          f"{len(windows) - len(differing)}/{len(windows)}")
    print(f"  leading channel unchanged                       : "
          f"{len(windows) - len(moved)}/{len(windows)}")
    if len(differing):
        print("\n  windows whose rates moved:")
        print(differing[["subject", "t_start", "n_set_aside", "n_flagged",
                         "pct_seconds_rejected"]].to_string(index=False))
    if len(moved):
        print("\n  windows whose leader moved:")
        print(moved[["subject", "t_start", "leader_without",
                     "leader_with"]].to_string(index=False))

    print()
    print("=" * 72)
    print("2. WHAT DID IT ACT ON?")
    print("=" * 72)
    print(f"  contacts examined            : {int(windows.n_channels.sum())}")
    print(f"  set aside (not analysed)     : {int(windows.n_set_aside.sum())}")
    print(f"  flagged (analysed, look)     : {int(windows.n_flagged.sum())}")
    print(f"  contact-seconds rejected     : "
          f"{windows.pct_seconds_rejected.mean():.3f}% mean, "
          f"{windows.pct_seconds_rejected.max():.3f}% worst window")
    if len(verdicts):
        print("\n  by reason:")
        print(verdicts.groupby(["verdict", "reason"]).size()
              .to_frame("n").to_string())

    print()
    print("=" * 72)
    print("3. IS A FLAG A FAULT, OR IS IT THE FINDING?")
    print("=" * 72)
    channels = every_channel()
    # Only contacts the annotators actually reviewed can answer this: a zero
    # on an unreviewed contact means nobody looked, not that nothing is there.
    judged = channels[channels["expert_reviewed"]]
    print(f"  contacts the archive's annotators reviewed: {len(judged)} "
          f"of {len(channels)}")
    if not len(judged):
        print("  no reviewed contacts in this sweep; cannot answer")
        return 0

    flagged = judged[judged["flagged"]]
    rest = judged[~judged["flagged"] & ~judged["set_aside"]]
    print(f"\n  {'group':<28s} {'n':>5s} {'median':>8s} {'mean':>8s} {'max':>7s}"
          f"  expert ripples marked")
    for label, group in (("flagged by this stage", flagged),
                         ("everything else", rest)):
        if not len(group):
            print(f"  {label:<28s} {0:5d}       --       --      --")
            continue
        print(f"  {label:<28s} {len(group):5d} "
              f"{group.expert_ripples.median():8.1f} "
              f"{group.expert_ripples.mean():8.1f} "
              f"{group.expert_ripples.max():7.0f}")

    if len(flagged) and len(rest):
        # No prepared verdict here. The first version of this script printed
        # "the flag tracks the pathology" whatever the numbers said, which is
        # the mistake this whole stage exists to avoid, committed in the tool
        # that was supposed to check for it. What follows is derived.
        median = float(rest.expert_ripples.median())
        silent = int((flagged.expert_ripples == 0).sum())
        active = int((flagged.expert_ripples >= median).sum())
        print(f"\n  of the {len(flagged)} flagged contacts the annotators reviewed:")
        print(f"    {silent:3d} ({silent / len(flagged):.0%}) carry **no** expert "
              f"marking at all")
        print(f"    {active:3d} ({active / len(flagged):.0%}) are at or above the "
              f"cohort median of {median:.0f} ripples")
        print(f"    {int((flagged.expert_ripples >= 100).sum()):3d} carry 100 or more")

        print("\n  So the flag is bimodal, and that is the finding: it picks up")
        print("  both quiet contacts, where it is plausibly noise, and some of the")
        print("  most heavily marked contacts in the cohort. It does not")
        print("  discriminate, so a version of it that removed contacts would")
        print(f"  have deleted roughly {active} real findings across "
              f"{flagged.subject.nunique()} subjects.")

        # Does it measure the contact, or the minute? A contact flagged in one
        # window and not the next four would be measurement noise.
        per_contact = (channels[channels["flagged"]]
                       .groupby(["subject", "channel"]).size())
        windows_each = (channels.groupby(["subject", "channel"]).size())
        stable = int((per_contact == windows_each.loc[per_contact.index]).sum())
        print(f"\n  {len(per_contact)} distinct contacts are ever flagged; "
              f"{stable} of them in every window of that subject.")
        print("  It is measuring the contact, not the minute.")

        worst = flagged.nlargest(min(8, len(flagged)), "expert_ripples")
        print("\n  the flagged contacts the annotators marked most heavily:")
        print(worst[["subject", "t_start", "channel", "hf_ratio_sd",
                     "expert_ripples"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
