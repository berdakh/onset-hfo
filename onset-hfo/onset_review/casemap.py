"""A case's combined contact map: interictal rate, ictal index, place, and the
clinician's onset zone, per channel. Qt-free.

Each analysed channel gets, side by side:

* its pooled **interictal** rate with its interval, whether it is tied with
  the busiest, and in how many segments (`onset_review.caseinterictal`);
* its **ictal** median Epileptogenicity Index and in how many seizures it
  reached the cutoff (`onset_review.caseictal`);
* its **template position** -- a contact's own, a bipolar pair's midpoint --
  and its contacts' probable atlas structures (`onset_hfo.case.electrodes`);
* whether the clinician put it in the **seizure onset zone** (either of its
  contacts; `Case.zones`).

`agreement` then says, for this patient, how the two measures and the zone
line up: the rank-based AUC of each measure for the zone's channels, whether
the top channel of each is in the zone, and the overlap of the channels each
singles out. One patient's agreement is an observation, not a validation;
the archive studies (`docs/OUTCOME.md`, `docs/ICTAL.md`) are where the
methods are tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["combined_table", "agreement", "draw_combined", "COMBINED_COLUMNS"]

COMBINED_COLUMNS = ("channel", "rate_per_min", "rate_ci_low", "rate_ci_high", "tied",
                    "segments_tied", "segments_analysed", "median_ei", "seizures_high",
                    "seizures", "x", "y", "z", "placed", "where", "soz")


def _where(frame: pd.DataFrame, channel: str) -> str:
    from onset_hfo.case.electrodes import contacts_of

    if not len(frame):
        return ""
    known = {str(n).upper(): (str(lab), str(how), mm) for n, lab, how, mm in
             zip(frame["name"], frame["label"], frame["label_how"], frame["label_mm"],
                 strict=True)}
    parts = []
    for contact in contacts_of(channel):
        if contact not in known:
            return ""
        label, how, _mm = known[contact]
        text = label if how == "in" else (f"near {label}" if how == "near" else how)
        if text and text not in parts:
            parts.append(text)
    return " / ".join(parts)


def combined_table(case) -> pd.DataFrame:
    """One row per channel analysed by either step."""
    from onset_hfo.case.electrodes import channel_positions, contacts_of, load_electrodes
    from onset_review.caseictal import latest_ictal
    from onset_review.caseinterictal import latest_result

    inter = latest_result(case)
    ictal = latest_ictal(case)
    channels: list[str] = []
    for table in ((inter.table if inter else None), (ictal.combined if ictal else None)):
        if table is not None and len(table):
            channels += [c for c in table["channel"].astype(str) if c not in channels]
    rows = pd.DataFrame({"channel": channels})
    if inter is not None and len(inter.table):
        keep = inter.table[["channel", "rate_per_min", "rate_ci_low", "rate_ci_high", "tied",
                            "segments_tied", "segments_analysed"]]
        rows = rows.merge(keep, on="channel", how="left")
    if ictal is not None and len(ictal.combined):
        rows = rows.merge(ictal.combined[["channel", "median_ei", "seizures_high",
                                          "seizures"]], on="channel", how="left")
    for column in COMBINED_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    electrodes = load_electrodes(case)
    where = channel_positions(electrodes, channels)
    rows[["x", "y", "z", "placed"]] = where[["x", "y", "z", "placed"]].to_numpy()
    rows["placed"] = rows["placed"].astype(bool)
    rows["where"] = [_where(electrodes, c) for c in channels]
    zone = set(case.zone("soz"))
    rows["soz"] = [any(p in zone for p in contacts_of(c)) for c in channels]
    rows["tied"] = rows["tied"].astype("boolean").fillna(False).astype(bool)
    for column in ("x", "y", "z", "rate_per_min", "median_ei"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    order = rows["rate_per_min"].rank(ascending=False).fillna(len(rows)) + \
        rows["median_ei"].rank(ascending=False).fillna(len(rows))
    return rows.assign(_order=order).sort_values(["_order", "channel"]).drop(
        columns="_order").reset_index(drop=True)[list(COMBINED_COLUMNS)]


def _auc(scores, positive) -> float:
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    ok = np.isfinite(scores)
    a, b = scores[ok & positive], scores[ok & ~positive]
    if not len(a) or not len(b):
        return float("nan")
    greater = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
    return float(greater / (len(a) * len(b)))


def agreement(table: pd.DataFrame, consistent: list[str] | None = None) -> dict:
    """How the interictal rate and the ictal index line up with the zone, for
    this patient. `consistent` is the ictal step's channels at the cutoff in at
    least half the seizures."""
    soz = table["soz"].astype(bool)
    out = {"n_channels": int(len(table)), "n_soz": int(soz.sum()), "measures": {}}
    picked = {"interictal": set(table.loc[table["tied"], "channel"]),
              "ictal": set(consistent or [])}
    for name, column in (("interictal", "rate_per_min"), ("ictal", "median_ei")):
        values = table[column]
        if values.notna().sum() == 0:
            continue
        top = table.loc[values.idxmax(), "channel"] if values.notna().any() else ""
        chosen = picked[name]
        out["measures"][name] = {
            "auc": _auc(values, soz) if soz.any() else float("nan"),
            "top": str(top), "top_in_soz": bool(table.set_index("channel").at[top, "soz"])
            if top else False,
            "singled_out": sorted(chosen),
            "singled_out_in_soz": int(sum(table.set_index("channel").at[c, "soz"]
                                          for c in chosen if c in set(table["channel"])))}
    both = picked["interictal"] & picked["ictal"]
    out["both"] = sorted(both)
    return out


def statement(table: pd.DataFrame, found: dict) -> str:
    parts = []
    measures = found.get("measures", {})
    names = {"interictal": "interictal rate", "ictal": "ictal index"}
    for key, m in measures.items():
        text = f"Highest {names[key]}: {m['top']}"
        if found["n_soz"]:
            text += " (in the marked onset zone)" if m["top_in_soz"] else \
                " (outside the marked onset zone)"
            if np.isfinite(m["auc"]):
                text += f"; AUC for the zone's channels {m['auc']:.2f}"
        parts.append(text + ".")
    if "interictal" in measures and "ictal" in measures:
        both = found["both"]
        parts.append(("Singled out by both: " + ", ".join(both[:10]) + ".") if both else
                     "No channel is singled out by both.")
    if not found["n_soz"]:
        parts.append("No onset zone is marked: tick its contacts to see the agreement.")
    placed = int(table["placed"].sum())
    parts.append(f"{placed} of {len(table)} channels placed on the template.")
    return " ".join(parts) if parts else "Nothing analysed yet."


def draw_combined(figure, table: pd.DataFrame, atlas=None, view: str = "top",
                  glass: bool = True, positions: str = "template") -> dict:
    """Left: interictal rate against ictal index, one dot per channel, filled
    when in the marked onset zone. Right: where those channels are on the
    template, sized by rate and shaded by index -- on nilearn's glass brain
    (left, above, right) when nilearn is installed and `glass`, otherwise seen
    from above or the side over the atlas brain's outline. `positions` says
    where the positions came from (`electrodes.positions_kind`), for the caption."""
    from onset_hfo.case.electrodes import CAPTIONS
    from onset_hfo.ictal import EI_CUTOFF
    from onset_review import templatebrain
    from onset_review.studycharts import GRID, MUTED, SERIES, _quiet

    figure.clear()
    use_glass = glass and templatebrain.available() and bool(table["placed"].any()) \
        if len(table) else False
    if use_glass:
        left = figure.add_axes([0.08, 0.15, 0.23, 0.66])
        right = None
    else:
        left = figure.add_subplot(1, 2, 1)
        right = figure.add_subplot(1, 2, 2)
    _quiet(left)
    left.grid(True, color=GRID, linewidth=0.8)
    both = table.dropna(subset=["rate_per_min", "median_ei"], how="all")
    rate = both["rate_per_min"].fillna(0.0)
    ei = both["median_ei"].fillna(0.0)
    soz = both["soz"].astype(bool)
    points = {}
    left.scatter(rate[~soz], ei[~soz], s=22, facecolors="white", edgecolors=SERIES[0],
                 linewidths=1.2, label="other channels")
    left.scatter(rate[soz], ei[soz], s=26, color=SERIES[1], label="marked onset zone")
    left.axhline(EI_CUTOFF, color=MUTED, linestyle="--", linewidth=0.9)
    for row in both.head(4).itertuples():
        left.annotate(row.channel, (row.rate_per_min if pd.notna(row.rate_per_min) else 0.0,
                                    row.median_ei if pd.notna(row.median_ei) else 0.0),
                      fontsize=7, color=MUTED, xytext=(4, 3), textcoords="offset points")
        points[row.channel] = row
    left.set_xlabel("interictal events/min (pooled)", fontsize=8, color=MUTED)
    left.set_ylabel("ictal index (median over seizures)", fontsize=8, color=MUTED)
    left.set_ylim(-0.03, 1.05)
    left.set_xlim(left=0)
    if soz.any():
        if use_glass:
            left.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(-0.05, 1.0),
                        ncol=1)
        else:
            left.legend(fontsize=7, frameon=False, loc="upper right",
                        bbox_to_anchor=(1.0, 1.12), ncol=2)
    if use_glass:
        templatebrain.draw_glass(figure, table, rect=(0.35, 0.02, 0.65, 0.86))
        figure.text(0.675, 0.93, templatebrain.glass_caption(positions), ha="center", va="top",
                    fontsize=7, color=MUTED)
        return points
    _quiet(right)
    right.set_aspect("equal")
    axes = (0, 1) if view == "top" else (1, 2)
    if atlas is not None:
        a, b, image = atlas.silhouette(2 if view == "top" else 0)
        right.contour(a, b, image.T, levels=[0.5], colors=[GRID], linewidths=1.2)
    placed = table[table["placed"]]
    if len(placed):
        size = 12 + 60 * (placed["rate_per_min"].fillna(0) /
                          max(float(placed["rate_per_min"].max() or 1.0), 1e-9))
        shade = placed["median_ei"].fillna(0.0).clip(0, 1)
        coords = placed[["x", "y", "z"]].to_numpy(float)
        right.scatter(coords[:, axes[0]], coords[:, axes[1]], s=size, c=shade, cmap="viridis",
                      vmin=0, vmax=1, edgecolors=[SERIES[1] if s else MUTED
                                                  for s in placed["soz"]],
                      linewidths=[1.4 if s else 0.4 for s in placed["soz"]])
    right.set_xlabel(("left ← x → right (mm)" if view == "top" else "back ← y → front (mm)"),
                     fontsize=8, color=MUTED)
    right.set_ylabel(("back ← y → front (mm)" if view == "top" else "down ← z → up (mm)"),
                     fontsize=8, color=MUTED)
    right.set_title(f"{CAPTIONS.get(positions, CAPTIONS['template'])}\n"
                    "size: rate · shade: index · ring: zone",
                    fontsize=8, color=MUTED)
    figure.tight_layout()
    return points
