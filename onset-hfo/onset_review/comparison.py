"""Two analyses side by side: what changed, and where. Qt-free.

The question after changing a setting -- bipolar to Laplacian, ripple to
fast ripple, a threshold -- is not "what does the new one say" but "what
moved". This lines the two up:

* **what differs** between the two requests, setting by setting;
* **the ranking**: every channel's rate and rank under each, how far it
  moved, Spearman's ρ over the channels both rank, how many of the top five
  they share, and the two tied sets;
* **the events**: per detector, the accepted events found by both, by the
  first only and by the second only, over the time both analysed;
* **the verdicts**: channels a reader judged differently in the two reads.

Two montages name channels differently -- a bipolar ``A1-A2`` is two
contacts, a Laplacian ``A1`` one -- so when the channels do not match, the
two are compared **by contact**: a contact's rate is that of the busiest
channel built from it, and two events are the same when they overlap in
time and share a contact. Which of the two was done is said in every
summary, because "the leader moved from A1-A2 to A2" is not a disagreement.

Nothing here is a new analysis: both sides are sessions the window built,
and every number is read from them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Comparison", "compare", "settings_differences", "contacts_of_channel",
           "to_markdown", "draw_rates", "TOLERANCE_S", "TOP"]

#: Events this close in time (seconds) still count as overlapping.
TOLERANCE_S = 0.02
#: How many leading channels "top" means.
TOP = 5


def contacts_of_channel(name: str) -> list[str]:
    """The contacts a channel is built from: ``A1-A2`` is A1 and A2."""
    return [part.strip().upper() for part in str(name).split("-") if part.strip()]


# -- what differs ------------------------------------------------------------------------------
_REQUEST_FIELDS = (
    ("recording", lambda r: r.label() if hasattr(r, "label") else ""),
    ("detectors", lambda r: ", ".join(r.detectors)),
    ("band", lambda r: r.band_label() if hasattr(r, "band_label") else r.band),
    ("threshold", lambda r: f"{r.threshold_sd:g} SD" if r.threshold_sd is not None
     else "each detector's default"),
    ("quality stage", lambda r: "on" if r.check_quality else "off"),
    ("spike check", lambda r: "on" if r.with_spikes else "off"),
    ("electrode file", lambda r: r.electrodes_path.name if r.electrodes_path else "none"),
)


def _plain(value) -> str:
    if value is None or value == () or value == "":
        return "–"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (tuple, list)):
        return ", ".join(":".join(str(v) for v in item) if isinstance(item, (tuple, list))
                         else str(item) for item in value) or "–"
    return f"{value:g}" if isinstance(value, float) else str(value)


def settings_differences(a, b) -> list[tuple[str, str, str]]:
    """(setting, under A, under B) for every setting the two requests differ in."""
    from onset_hfo.config import PreprocessConfig
    from onset_hfo.preprocess import effective_reference

    out = []
    for name, read in _REQUEST_FIELDS:
        try:
            left, right = read(a), read(b)
        except Exception:       # noqa: BLE001 - a field one side lacks is not a difference
            continue
        if left != right:
            out.append((name, str(left), str(right)))
    pa = a.preprocess or PreprocessConfig()
    pb = b.preprocess or PreprocessConfig()
    try:
        ra, rb = effective_reference(pa), effective_reference(pb)
    except ValueError:
        ra, rb = str(pa.reference), str(pb.reference)
    if ra != rb:
        out.append(("reference", ra, rb))
    skip = {"reference", "bipolar", "average_reference"}
    for f in dataclasses.fields(PreprocessConfig):
        if f.name in skip:
            continue
        left, right = getattr(pa, f.name), getattr(pb, f.name)
        if left != right:
            out.append((f.name.replace("_", " "), _plain(left), _plain(right)))
    return out


# -- the ranking -------------------------------------------------------------------------------
def _rates(session, by_contact: bool) -> pd.Series:
    findings = session.findings
    if findings is None or findings.empty:
        return pd.Series(dtype=float)
    rates = pd.Series(findings["rate_per_min"].astype(float).to_numpy(),
                      index=findings["channel"].astype(str))
    if not by_contact:
        return rates
    out: dict[str, float] = {}
    for channel, rate in rates.items():
        for contact in contacts_of_channel(channel):
            if not np.isfinite(out.get(contact, np.nan)) or rate > out[contact]:
                out[contact] = float(rate)
    return pd.Series(out, dtype=float)


def _tied(session, by_contact: bool) -> set[str]:
    names = [str(c) for c in (session.candidates or [])]
    if not by_contact:
        return set(names)
    return {contact for name in names for contact in contacts_of_channel(name)}


def _leader(session) -> str:
    leader = session.leader or {}
    return str(leader.get("leader") or "") if leader.get("available", True) else ""


# -- the events --------------------------------------------------------------------------------
def _keys(event, by_contact: bool) -> set[str]:
    return set(contacts_of_channel(event.channel)) if by_contact else {str(event.channel)}


def _match(a: list, b: list, by_contact: bool, tolerance: float = TOLERANCE_S) -> int:
    """How many events of `a` have a partner in `b`: overlapping in time and
    on the same channel (or sharing a contact), one to one, nearest first."""
    order_b = sorted(range(len(b)), key=lambda j: b[j].start)
    starts = np.array([b[j].start for j in order_b]) if order_b else np.array([])
    used: set[int] = set()
    longest = max((e.stop - e.start for e in b), default=0.0)
    matched = 0
    for event in sorted(a, key=lambda e: e.start):
        keys = _keys(event, by_contact)
        lo = np.searchsorted(starts, event.start - longest - tolerance, side="left")
        hi = np.searchsorted(starts, event.stop + tolerance, side="right")
        best, gap = None, None
        for position in range(lo, hi):
            j = order_b[position]
            other = b[j]
            if j in used or other.stop < event.start - tolerance \
                    or other.start > event.stop + tolerance:
                continue
            if not keys & _keys(other, by_contact):
                continue
            distance = abs((other.start + other.stop) / 2 - (event.start + event.stop) / 2)
            if gap is None or distance < gap:
                best, gap = j, distance
        if best is not None:
            used.add(best)
            matched += 1
    return matched


# -- the comparison ----------------------------------------------------------------------------
@dataclass
class Comparison:
    """Two analyses lined up. `unit` is "channel" or "contact"."""

    a_label: str
    b_label: str
    unit: str
    settings: list = field(default_factory=list)
    ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    rho: float = float("nan")
    n_ranked: int = 0
    top_shared: int = 0
    leader_a: str = ""
    leader_b: str = ""
    tied_a: set = field(default_factory=set)
    tied_b: set = field(default_factory=set)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap: tuple = (float("nan"), float("nan"))
    verdicts: pd.DataFrame = field(default_factory=pd.DataFrame)
    same_recording: bool = True
    sentences: list = field(default_factory=list)


def compare(a, b, a_label: str = "A", b_label: str = "B") -> Comparison:
    """Line up two review sessions (`onset_review.session.ReviewSession`)."""
    from scipy.stats import spearmanr

    names_a = set(a.findings["channel"].astype(str)) if not a.findings.empty else set()
    names_b = set(b.findings["channel"].astype(str)) if not b.findings.empty else set()
    by_contact = names_a != names_b and bool(names_a) and bool(names_b)
    unit = "contact" if by_contact else "channel"
    rates_a, rates_b = _rates(a, by_contact), _rates(b, by_contact)
    units = sorted(set(rates_a.index) | set(rates_b.index))
    frame = pd.DataFrame({unit: units,
                          "rate_a": [rates_a.get(u, np.nan) for u in units],
                          "rate_b": [rates_b.get(u, np.nan) for u in units]})
    frame["rank_a"] = frame["rate_a"].rank(ascending=False, method="min")
    frame["rank_b"] = frame["rate_b"].rank(ascending=False, method="min")
    frame["moved"] = frame["rank_a"] - frame["rank_b"]     # positive: higher under B
    tied_a, tied_b = _tied(a, by_contact), _tied(b, by_contact)
    frame["tied_a"] = frame[unit].isin(tied_a)
    frame["tied_b"] = frame[unit].isin(tied_b)
    frame = frame.sort_values(["rank_a", "rank_b"], na_position="last").reset_index(drop=True)
    both = frame.dropna(subset=["rate_a", "rate_b"])
    rho = float("nan")
    if len(both) >= 3 and both["rate_a"].nunique() > 1 and both["rate_b"].nunique() > 1:
        rho = float(spearmanr(both["rate_a"], both["rate_b"]).statistic)
    top_a = set(frame.nsmallest(TOP, "rank_a")[unit]) if frame["rank_a"].notna().any() else set()
    top_b = set(frame.nsmallest(TOP, "rank_b")[unit]) if frame["rank_b"].notna().any() else set()

    start = max(a.span[0], b.span[0])
    stop = min(a.span[1], b.span[1])
    rows = []
    detectors = [d for d in a.request.detectors if d in b.request.detectors]
    for detector in detectors:
        ea = [e for e in a.events if e.accepted and e.detector == detector
              and start <= (e.start + e.stop) / 2 < stop]
        eb = [e for e in b.events if e.accepted and e.detector == detector
              and start <= (e.start + e.stop) / 2 < stop]
        shared = _match(ea, eb, by_contact)
        rows.append({"detector": detector, "in_a": len(ea), "in_b": len(eb), "both": shared,
                     "only_a": len(ea) - shared, "only_b": len(eb) - shared})
    events = pd.DataFrame(rows, columns=["detector", "in_a", "in_b", "both", "only_a",
                                         "only_b"])

    verdicts = []
    read_a = getattr(a.read, "channels", {}) or {}
    read_b = getattr(b.read, "channels", {}) or {}
    for channel in sorted(set(read_a) | set(read_b)):
        va = getattr(read_a.get(channel), "verdict", "")
        vb = getattr(read_b.get(channel), "verdict", "")
        if va != vb:
            verdicts.append({"channel": channel, "verdict_a": va or "–", "verdict_b": vb or "–"})
    verdicts = pd.DataFrame(verdicts, columns=["channel", "verdict_a", "verdict_b"])

    same = _recording_of(a.request) == _recording_of(b.request)
    result = Comparison(a_label=a_label, b_label=b_label, unit=unit,
                        settings=settings_differences(a.request, b.request), ranking=frame,
                        rho=rho, n_ranked=len(both), top_shared=len(top_a & top_b),
                        leader_a=_leader(a), leader_b=_leader(b), tied_a=tied_a,
                        tied_b=tied_b, events=events, overlap=(start, stop),
                        verdicts=verdicts, same_recording=same)
    result.sentences = _sentences(result, a, b)
    return result


def _recording_of(request) -> tuple:
    if getattr(request, "path", None):
        return ("file", str(request.path))
    return (request.dataset, request.subject, request.run, getattr(request, "task", None))


def _sentences(c: Comparison, a, b) -> list[str]:
    out = []
    if c.settings:
        out.append("They differ in " + "; ".join(f"{name} ({left} → {right})"
                                                  for name, left, right in c.settings) + ".")
    else:
        out.append("The two were analysed with the same settings.")
    if not c.same_recording:
        out.append("They are different recordings: channels are lined up by name, which is "
                   "only meaningful when the names mean the same contacts.")
    if c.unit == "contact":
        out.append("Their channels are named differently (different montages), so they are "
                   "compared by contact: a contact's rate is that of the busiest channel built "
                   "from it, and two events are the same when they overlap and share a "
                   "contact.")

    def stand(session) -> str:
        leader = session.leader or {}
        return "stands out" if leader.get("distinguishable") else "is tied with others"

    if c.leader_a or c.leader_b:
        same = (set(contacts_of_channel(c.leader_a)) & set(contacts_of_channel(c.leader_b))
                if c.unit == "contact" else {c.leader_a} & {c.leader_b})
        out.append(f"Busiest channel: {c.leader_a or '–'} under {c.a_label} ({stand(a)}), "
                   f"{c.leader_b or '–'} under {c.b_label} ({stand(b)})"
                   + ("; the same place." if same else "; a different place."))
    if np.isfinite(c.rho):
        out.append(f"The rankings agree with Spearman ρ {c.rho:.2f} over the {c.n_ranked} "
                   f"{c.unit}s both rank; {c.top_shared} of the top {TOP} are shared.")
    shared = len(c.tied_a & c.tied_b)
    out.append(f"Tied with the busiest: {len(c.tied_a)} {c.unit}(s) under {c.a_label}, "
               f"{len(c.tied_b)} under {c.b_label}, {shared} in both.")
    for row in c.events.itertuples():
        out.append(f"{row.detector}: {row.both} accepted event(s) found by both, {row.only_a} "
                   f"only under {c.a_label}, {row.only_b} only under {c.b_label}.")
    if np.isfinite(c.overlap[0]) and c.overlap[1] <= c.overlap[0]:
        out.append("The two analysed no time in common, so their events cannot be matched.")
    if len(c.verdicts):
        out.append(f"{len(c.verdicts)} channel verdict(s) differ between the two reads.")
    return out


def to_markdown(c: Comparison, a_request=None, b_request=None) -> str:
    """The comparison as a Markdown page: what each was, what moved."""
    lines = [f"# Comparison — {c.a_label} and {c.b_label}", ""]
    if a_request is not None and b_request is not None:
        lines += [f"- **{c.a_label}**: {a_request.label()}",
                  f"- **{c.b_label}**: {b_request.label()}", ""]
    lines += [*c.sentences, ""]
    if c.settings:
        lines += ["## What differs", "", f"| setting | {c.a_label} | {c.b_label} |",
                  "|---|---|---|"]
        lines += [f"| {name} | {left} | {right} |" for name, left, right in c.settings]
        lines.append("")
    lines += [f"## Ranking, by {c.unit}", "",
              f"| {c.unit} | rank {c.a_label} | rank {c.b_label} | rate {c.a_label} /min | "
              f"rate {c.b_label} /min | moved |", "|---|---:|---:|---:|---:|---:|"]

    def num(value, fmt="{:g}"):
        return fmt.format(value) if isinstance(value, (int, float)) and np.isfinite(value) \
            else "–"

    for row in c.ranking.itertuples():
        unit = getattr(row, c.unit)
        lines.append(f"| {unit} | {num(row.rank_a)} | {num(row.rank_b)} | "
                     f"{num(row.rate_a, '{:.1f}')} | {num(row.rate_b, '{:.1f}')} | "
                     f"{num(row.moved, '{:+g}')} |")
    if len(c.events):
        lines += ["", "## Accepted events", "",
                  f"| detector | {c.a_label} | {c.b_label} | both | only {c.a_label} | "
                  f"only {c.b_label} |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {r.detector} | {r.in_a} | {r.in_b} | {r.both} | {r.only_a} | "
                  f"{r.only_b} |" for r in c.events.itertuples()]
    if len(c.verdicts):
        lines += ["", "## Channel verdicts that differ", "",
                  f"| channel | {c.a_label} | {c.b_label} |", "|---|---|---|"]
        lines += [f"| {r.channel} | {r.verdict_a} | {r.verdict_b} |"
                  for r in c.verdicts.itertuples()]
    return "\n".join(lines) + "\n"


def draw_rates(figure, axis, c: Comparison) -> dict:
    """Each channel's (or contact's) rate under A against under B: on the
    diagonal it did not move. Tied-with-the-busiest under either is filled."""
    from onset_review.studycharts import GRID, MARK, MUTED, SERIES, _quiet

    _quiet(axis)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    both = c.ranking.dropna(subset=["rate_a", "rate_b"])
    top = float(max(both["rate_a"].max(), both["rate_b"].max(), 1.0)) * 1.08 \
        if len(both) else 1.0
    axis.plot([0, top], [0, top], color=GRID, linewidth=1.2, zorder=1)
    tied = both["tied_a"] | both["tied_b"]
    points = axis.scatter(both["rate_a"], both["rate_b"], s=30,
                          c=[SERIES[0] if t else "white" for t in tied],
                          edgecolors=SERIES[0], linewidths=1.2, zorder=3)
    for row in both[tied].itertuples():
        axis.annotate(getattr(row, c.unit), (row.rate_a, row.rate_b), xytext=(4, 3),
                      textcoords="offset points", fontsize=7, color=MARK)
    axis.set_xlim(0, top)
    axis.set_ylim(0, top)
    axis.set_xlabel(f"events per minute under {c.a_label}", fontsize=8, color=MUTED)
    axis.set_ylabel(f"under {c.b_label}", fontsize=8, color=MUTED)
    figure.tight_layout()
    return {"points": points}
