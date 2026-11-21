"""Rebuild the whole-run outcome tables from the committed per-channel counts.

The outcome study (`onset_hfo.outcome.outcome_study`) detects events on the
whole 300 s run of every ds003498 patient, which takes the archive and about
ninety minutes. Everything after detection -- the per-patient metrics and the
group comparison -- is a function of the per-channel event counts, and those
are committed in ``data/outcome/channels.csv.gz``. This script reruns only
that second half, with the code in this checkout, so a change to how a metric
is computed (the rate interval, for instance) can be carried into
``data/outcome/subjects.csv`` and ``data/stability/outcome_groups_300s.csv``
without the archive.

    python scripts/rederive_outcome_tables.py            # rewrite both files
    python scripts/rederive_outcome_tables.py --check    # exit 1 if they would change

It refuses to write unless it first reproduces every column that the change
cannot touch (event shares, the top channel, the top three) exactly, which is
the evidence that the counts it starts from are the ones the study used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from onset_hfo.outcome import (  # noqa: E402
    _candidate_metrics,
    _share_in_resection,
    _top_channel_resected,
    _top_k_resected,
    compare_groups,
)

OUTCOME = HERE / "data" / "outcome"
GROUPS = HERE / "data" / "stability" / "outcome_groups_300s.csv"
UNCHANGED = ["n_channels", "n_resected_channels", "n_events", "n_events_resected",
             "n_events_partial", "n_events_spared", "share_in_rz",
             "share_in_rz_incl_partial", "top_channel_resected", "top3_resected"]


def rederive() -> tuple[pd.DataFrame, pd.DataFrame]:
    channels = pd.read_csv(OUTCOME / "channels.csv.gz")
    committed = pd.read_csv(OUTCOME / "subjects.csv")
    durations = pd.read_csv(OUTCOME / "recordings.csv").set_index("subject")["duration_s"]
    sources = [c[: -len("_events")] for c in channels.columns if c.endswith("_events")]

    rows = []
    ties_at_third: list[bool] = []
    for _, ref in committed.iterrows():
        part = channels[(channels["subject"] == ref["subject"]) & (channels["band"] == ref["band"])]
        # outcome_subject drops eloquent channels from both scopes (drop_eloquent=True).
        part = part[~part["eloquent"].astype(bool)]
        if ref["scope"] == "reviewed":
            part = part[part["reviewed"].astype(bool)]
        part = part.set_index("channel")
        if ref["source"] not in sources:
            raise SystemExit(f"no {ref['source']}_events column in channels.csv.gz")
        rates = part[f"{ref['source']}_events"].astype(float)
        zones = part["zone"]
        ordered = rates.sort_values(ascending=False).to_numpy()
        ties_at_third.append(len(ordered) > 3 and ordered[2] == ordered[3])
        rows.append({
            **{k: ref[k] for k in ("subject", "band", "scope", "source", "threshold_sd")},
            "n_channels": len(part),
            "n_resected_channels": int((zones == "resected").sum()),
            **_share_in_resection(rates, zones),
            "top_channel_resected": _top_channel_resected(rates, zones),
            "top3_resected": _top_k_resected(rates, zones, k=3),
            **_candidate_metrics(rates, zones, float(durations[ref["subject"]]) / 60.0),
        })
    subjects = pd.DataFrame(rows)[committed.columns]

    for column in UNCHANGED:
        a, b = committed[column].to_numpy(float), subjects[column].to_numpy(float)
        same = np.isclose(a, b, equal_nan=True)
        if column == "top3_resected":
            # Channels with equal counts keep the order the study saw them in,
            # which channels.csv.gz does not record; a tie at third place can
            # therefore pick a different channel here. Only those rows may differ.
            same |= np.asarray(ties_at_third, dtype=bool)
        if not same.all():
            raise SystemExit(f"{column} does not reproduce from channels.csv.gz; refusing to write")
        # Nothing the interval touches lives in these columns: keep the study's values.
        subjects[column] = committed[column]

    participants = pd.read_csv(OUTCOME / "participants.csv")
    groups = compare_groups(subjects, participants)
    return subjects, groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report whether the committed files would change, write nothing")
    args = parser.parse_args()
    subjects, groups = rederive()
    old_subjects = pd.read_csv(OUTCOME / "subjects.csv")
    old_groups = pd.read_csv(GROUPS)
    same_subjects = subjects.round(9).equals(old_subjects.round(9))
    groups = groups[old_groups.columns]
    same_groups = groups.round(9).reset_index(drop=True).equals(old_groups.round(9))
    print(f"subjects.csv: {'unchanged' if same_subjects else 'would change'}")
    print(f"outcome_groups_300s.csv: {'unchanged' if same_groups else 'would change'}")
    if args.check:
        return 0 if same_subjects and same_groups else 1
    subjects.to_csv(OUTCOME / "subjects.csv", index=False)
    groups.to_csv(GROUPS, index=False)
    # What chance alone gives each group, from the same per-patient table.
    from onset_hfo.outcome import chance_baseline

    chance_baseline(subjects, pd.read_csv(OUTCOME / "participants.csv")).to_csv(
        OUTCOME / "chance_baseline.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
