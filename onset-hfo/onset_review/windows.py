"""The other windows of the same patient, for questions across them.

"Did the leader change between the first and the second minute?" is a
question about two analyses, and the assistant could only see one. This
module finds the other windows of the same recording already on this
machine (the launcher's cache), analyses one on request with the settings
of the window on screen -- the same band, detector, threshold and
preprocessing, so the two are comparable -- and compares the leaders.

Qt-free. The analysis is :func:`onset_review.session.load_session`, the same
path the window itself was built by, offline, so it costs what opening the
window cost: seconds to half a minute. A result is kept in memory for the
rest of the session and on disk beside the settings, keyed by the recording
and the analysis settings, so the question asked twice costs once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

__all__ = ["other_windows", "analyse_window", "compare", "summarise", "window_key"]

_CACHE: dict[str, dict] = {}


def _cache_dir() -> Path | None:
    try:
        from onset_review.assistant_config import config_path

        return config_path().parent / "windows"
    except Exception:       # noqa: BLE001 - no settings dir, no disk cache
        return None


def window_key(request) -> str:
    """The recording window plus everything that changes its numbers."""
    from onset_review.adjudication import window_id

    settings = {
        "detectors": list(request.detectors), "band": request.band,
        "threshold_sd": request.threshold_sd, "with_spikes": request.with_spikes,
        "check_quality": request.check_quality, "keep": list(request.keep_channels),
        "preprocess": request.preprocess_label,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str)
                            .encode()).hexdigest()[:10]
    return f"{window_id(request)}-{digest}"


def other_windows(session, cache_dir=None, lister=None) -> list[dict]:
    """The cached windows of this recording other than the one on screen,
    earliest first, each saying whether it has been analysed already."""
    from onset_review.session import cached_windows

    request = session.request
    if getattr(request, "path", None) is not None:
        return []                       # an imported file has no cache of windows
    frame = (lister or cached_windows)(cache_dir)
    if frame is None or frame.empty:
        return []
    span = request.span()
    rows = []
    for row in frame.itertuples():
        if (str(row.dataset) != str(request.dataset) or str(row.subject) != str(request.subject)
                or str(row.run) != str(request.run)):
            continue
        start, stop = float(row.t_start), float(row.t_stop)
        if abs(start - span[0]) < 1e-6 and abs(stop - span[1]) < 1e-6:
            continue
        other = dataclasses.replace(request, t_start=start, t_stop=stop,
                                    span_start=None, span_stop=None)
        rows.append({"t_start": start, "t_stop": stop, "duration_s": round(stop - start, 1),
                     "sfreq": float(row.sfreq),
                     "analysed": window_key(other) in _CACHE or _on_disk(window_key(other))})
    return sorted(rows, key=lambda r: r["t_start"])


def _on_disk(key: str) -> bool:
    root = _cache_dir()
    return bool(root) and (root / f"{key}.json").exists()


def summarise(session, top: int = 5) -> dict:
    """What one window says: its span, leader, tied set and leading rates."""
    findings = session.findings
    ranked = findings.sort_values("rank") if not findings.empty else findings
    leading = [{"channel": str(r.channel), "rank": int(r.rank),
                "rate_per_min": round(float(r.rate_per_min), 2),
                "n_events": int(r.n_events)}
               for r in ranked.head(top).itertuples()] if not ranked.empty else []
    span = session.span
    return {
        "window_s": [float(span[0]), float(span[1])],
        "leader": session.leader.get("leader") if session.leader.get("available") else None,
        "leader_stands_out": session.leader.get("distinguishable"),
        "tied_with_leader": list(session.candidates),
        "leading": leading,
        "n_accepted": int(sum(1 for e in session.events if e.accepted)),
        "channels_analysed": int(findings["rank"].notna().sum()) if "rank" in findings else 0,
    }


def analyse_window(session, t_start: float, t_stop: float, cache_dir=None,
                   loader=None, top: int = 5) -> dict:
    """Analyse another window of this recording with this window's settings
    and return its summary. Cached in memory and on disk."""
    request = dataclasses.replace(session.request, t_start=float(t_start),
                                  t_stop=float(t_stop), span_start=None, span_stop=None)
    key = window_key(request)
    if key in _CACHE:
        return dict(_CACHE[key], cached=True)
    root = _cache_dir()
    if root is not None and (root / f"{key}.json").exists():
        try:
            summary = json.loads((root / f"{key}.json").read_text(encoding="utf-8"))
            _CACHE[key] = summary
            return dict(summary, cached=True)
        except (OSError, ValueError):
            pass
    from onset_review.session import load_session

    other = (loader or load_session)(request, cache_dir)
    summary = summarise(other, top=top)
    _CACHE[key] = summary
    if root is not None:
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{key}.json").write_text(json.dumps(summary), encoding="utf-8")
        except OSError:
            pass
    return dict(summary, cached=False)


def compare(here: dict, there: dict) -> dict:
    """Two summaries side by side: did the leader change, who moved."""
    rank_here = {row["channel"]: row["rank"] for row in here["leading"]}
    rank_there = {row["channel"]: row["rank"] for row in there["leading"]}
    rate_here = {row["channel"]: row["rate_per_min"] for row in here["leading"]}
    rate_there = {row["channel"]: row["rate_per_min"] for row in there["leading"]}
    channels = sorted(set(rank_here) | set(rank_there),
                      key=lambda c: (min(rank_here.get(c, 99), rank_there.get(c, 99)), c))
    moves = [{"channel": c, "rank_this_window": rank_here.get(c),
              "rank_other_window": rank_there.get(c),
              "rate_this_window": rate_here.get(c), "rate_other_window": rate_there.get(c)}
             for c in channels]
    same_leader = (here["leader"] is not None and here["leader"] == there["leader"])
    in_tied = (here["leader"] is not None and here["leader"] in there["tied_with_leader"])
    return {
        "this_window_s": here["window_s"], "other_window_s": there["window_s"],
        "leader_this_window": here["leader"], "leader_other_window": there["leader"],
        "leader_changed": not same_leader,
        "this_leader_tied_with_other_leader": in_tied,
        "tied_this_window": here["tied_with_leader"],
        "tied_other_window": there["tied_with_leader"],
        "leading_channels": moves,
        "note": ("A leader that changes between windows of the same recording is the "
                 "usual case, not a finding: rates vary minute to minute, and two "
                 "channels tied in one window may swap in the next. Compare the tied "
                 "sets, not the single leader."),
    }
