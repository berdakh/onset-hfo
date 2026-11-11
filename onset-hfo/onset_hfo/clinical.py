"""Which tissue was removed, and what happened to the patient afterwards.

Everything else in this package asks *where are the HFOs?*. This module
supplies the two facts needed to ask the question that actually matters --
**did removing the tissue that generated them stop the seizures?**

Two sidecars in ``ds003498`` carry them:

``sourcedata/clinical_ch_sheet_zurich.xlsx``
    One row per subject, with ``rz`` -- the **resected zone**, written as
    free-text contact ranges (``"ahr1-4, ar1-4, phr1-4"``) -- and
    ``excluded``, the contacts the original study dropped because electrical
    stimulation of them evoked a motor or language response (eloquent
    cortex). The dataset README quotes the paper on that exclusion.

``participants.tsv``
    ``outcome`` (``S`` seizure-free / ``F`` recurrence), the ILAE class,
    months of follow-up, and whether the epilepsy was temporal (TLE) or
    extratemporal (ETE).

Why this is fiddly, and why it is a module rather than three lines in a
notebook:

1. The sheet names **contacts** (``AHR1``), the analysis works on **bipolar
   channels** (``AHR1-AHR2``). A bipolar channel sits between two contacts,
   so it can be fully inside the resection, fully outside, or -- at the
   resection margin -- straddling it. Collapsing that third case into either
   of the others is a silent decision about the hardest channels in the
   dataset, so :func:`classify_channels` keeps it as its own label,
   ``partial``, and the analysis decides explicitly what to do with it.
2. Recorded channels are a subset of implanted contacts, and the sheet lists
   implanted ones. A resected contact that was never recorded cannot be
   scored, and its absence has to be *reported*, not absorbed: that is what
   ``coverage`` is for.
3. The sheet has a typo (subject 15's excluded list says ``1ll22-24`` where
   every other token on the row says ``tll``). Nothing here repairs it
   quietly; :class:`Resection` carries an ``unparsed`` field and
   :func:`resection_map` prints a warning.

Nothing in this module is specific to Zurich apart from the file path, which
lives in :class:`~onset_hfo.config.DatasetSpec`. A dataset without
``clinical_sheet`` simply has no resected zone, and the functions say so.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from onset_hfo.config import DATA_CACHE, DatasetSpec
from onset_hfo.datasets import (
    _dataset_url,
    _http_get,
    _spec,
    archive_electrode_names,
    expand_contact_ranges,
    read_tsv_text,
    rename_contacts,
)

__all__ = [
    "Resection",
    "fetch_clinical_sheet",
    "fetch_participants",
    "resection_map",
    "classify_channels",
    "ZONES",
]

#: The three positions a bipolar channel can occupy relative to a resection.
#: Ordered from inside to outside, which is the order they are reported in.
ZONES = ("resected", "partial", "spared")


@dataclass(frozen=True)
class Resection:
    """The resected zone for one subject, as contacts rather than free text."""

    subject: str                       #: BIDS id, e.g. ``"sub-01"``
    resected: tuple[str, ...]          #: contacts inside the resection, uppercase
    eloquent: tuple[str, ...]          #: contacts excluded for stimulation responses
    rz_text: str = ""                  #: the sheet cell, verbatim, for auditing
    excluded_text: str = ""
    unparsed: tuple[str, ...] = ()     #: chunks the parser could not expand

    def __len__(self) -> int:
        return len(self.resected)

    def coverage(self, ch_names: list[str]) -> dict:
        """How much of the resected zone these recorded contacts can see.

        ``recorded`` over ``listed`` is the honest denominator for every
        claim downstream: if only half the resected contacts were recorded,
        "HFOs inside the resection" was measured on half a resection.
        """
        contacts = {c.upper() for name in ch_names for c in name.split("-")}
        seen = [c for c in self.resected if c in contacts]
        return {
            "subject": self.subject,
            "rz_contacts_listed": len(self.resected),
            "rz_contacts_recorded": len(seen),
            "rz_coverage": (len(seen) / len(self.resected)) if self.resected else float("nan"),
            "missing": ", ".join(c for c in self.resected if c not in contacts),
        }


# --------------------------------------------------------------------------
# Fetching the sidecars
# --------------------------------------------------------------------------


def _cache_path(spec: DatasetSpec, name: str, cache_dir: str | Path | None) -> Path:
    directory = Path(cache_dir or DATA_CACHE) / spec.dataset_id / "clinical"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def fetch_clinical_sheet(dataset: str | DatasetSpec | None = "ds003498",
                         cache_dir: str | Path | None = None,
                         force: bool = False) -> pd.DataFrame:
    """The clinical sheet as a frame with ``subject``, ``rz`` and ``excluded``.

    Cached next to the signal slices, because it is 9 kB and the network is
    the slowest part of every notebook. ``subject`` comes back as a BIDS id
    (``sub-01``), not the sheet's bare integer.

    Raises ``ValueError`` if the dataset ships no clinical sheet.
    """
    spec = _spec(dataset)
    if not spec.clinical_sheet:
        raise ValueError(
            f"{spec.dataset_id} ships no clinical sheet, so the resected zone is unknown. "
            "Only ds003498 carries one.")
    path = _cache_path(spec, Path(spec.clinical_sheet).name, cache_dir)
    if force or not path.exists():
        path.write_bytes(_http_get(_dataset_url(spec, *spec.clinical_sheet.split("/"))))
    try:
        frame = pd.read_excel(io.BytesIO(path.read_bytes()))
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "reading the clinical sheet needs openpyxl: pip install openpyxl") from exc
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = {"subject", "rz"} - set(frame.columns)
    if missing:
        raise ValueError(f"clinical sheet is missing column(s) {sorted(missing)}")
    frame["subject"] = [_bids_subject(v) for v in frame["subject"]]
    if "excluded" not in frame.columns:
        frame["excluded"] = None
    return frame[["subject", "rz", "excluded"]]


def fetch_participants(dataset: str | DatasetSpec | None = "ds003498",
                       cache_dir: str | Path | None = None,
                       force: bool = False) -> pd.DataFrame:
    """``participants.tsv``, with ``participant_id`` renamed to ``subject``.

    For ``ds003498`` this carries the reference standard the outcome study
    needs: ``outcome`` is ``S`` (seizure-free) or ``F`` (recurrence), from
    Table 1 of the source publication, with ILAE class and follow-up length
    beside it.
    """
    spec = _spec(dataset)
    path = _cache_path(spec, "participants.tsv", cache_dir)
    if force or not path.exists():
        path.write_bytes(_http_get(_dataset_url(spec, "participants.tsv")))
    frame = read_tsv_text(path.read_text(encoding="utf-8", errors="replace"))
    if frame is None or not len(frame):
        raise ValueError(f"{spec.dataset_id}/participants.tsv could not be parsed")
    frame = frame.rename(columns={"participant_id": "subject"})
    frame["subject"] = [_bids_subject(v) for v in frame["subject"]]
    return frame


def _bids_subject(value) -> str:
    """``1``, ``"1"``, ``"01"`` and ``"sub-01"`` all mean ``"sub-01"``."""
    text = str(value).strip()
    if text.lower().startswith("sub-"):
        return "sub-" + text[4:].strip()
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return f"sub-{text}"


# --------------------------------------------------------------------------
# Sheet -> contacts
# --------------------------------------------------------------------------


def resection_map(dataset: str | DatasetSpec | None = "ds003498",
                  cache_dir: str | Path | None = None,
                  verbose: bool = True) -> dict[str, Resection]:
    """Parse the whole clinical sheet into :class:`Resection` objects.

    Keyed by BIDS subject id. Subjects whose ``rz`` cell is empty are still
    returned, with an empty ``resected`` tuple, so a caller that iterates the
    map sees them and can exclude them deliberately.
    """
    sheet = fetch_clinical_sheet(dataset, cache_dir=cache_dir)
    spec = _spec(dataset)
    out: dict[str, Resection] = {}
    for _, row in sheet.iterrows():
        rz_text = "" if pd.isna(row["rz"]) else str(row["rz"])
        ex_text = "" if pd.isna(row["excluded"]) else str(row["excluded"])
        resected, bad_rz = expand_contact_ranges(rz_text)
        eloquent, bad_ex = expand_contact_ranges(ex_text)
        # The sheet uses the markings' electrode names, which in a few
        # patients are not the recording's (`datasets.archive_electrode_names`).
        names = archive_electrode_names(spec, row["subject"])
        record = Resection(
            subject=row["subject"],
            resected=tuple(dict.fromkeys(rename_contacts(c.upper(), names) for c in resected)),
            eloquent=tuple(dict.fromkeys(rename_contacts(c.upper(), names) for c in eloquent)),
            rz_text=rz_text, excluded_text=ex_text,
            unparsed=tuple(bad_rz + bad_ex),
        )
        if verbose and record.unparsed:
            print(f"[onset-hfo] {record.subject}: could not parse "
                  f"{', '.join(record.unparsed)!r} in the clinical sheet "
                  "(left out of the contact lists)")
        out[record.subject] = record
    return out


def classify_channels(ch_names: list[str], resection: Resection) -> pd.DataFrame:
    """Label each (bipolar) channel ``resected``, ``partial`` or ``spared``.

    A channel is

    ``resected``
        both of its contacts were removed -- the signal it recorded came from
        tissue that no longer exists;
    ``partial``
        exactly one contact was removed. These sit on the resection margin.
        Counting them as resected inflates the "we got it all" number;
        counting them as spared inflates the opposite one. They are reported
        separately so that whatever an analysis does with them is visible;
    ``spared``
        neither contact was removed.

    ``eloquent`` is a separate boolean column, not a fourth zone, because it
    is a different kind of fact: the original study did not score those
    contacts at all, and an analysis may want to drop them regardless of
    which side of the resection they were on.
    """
    rz, elo = set(resection.resected), set(resection.eloquent)
    rows = []
    for name in ch_names:
        contacts = [c.upper() for c in str(name).split("-") if c]
        inside = sum(1 for c in contacts if c in rz)
        zone = ("resected" if inside == len(contacts) and inside
                else "partial" if inside else "spared")
        rows.append({
            "channel": name,
            "contact_a": contacts[0] if contacts else "",
            "contact_b": contacts[1] if len(contacts) > 1 else "",
            "n_contacts": len(contacts),
            "n_resected_contacts": inside,
            "zone": zone,
            "eloquent": any(c in elo for c in contacts),
        })
    return pd.DataFrame(rows, columns=["channel", "contact_a", "contact_b", "n_contacts",
                                       "n_resected_contacts", "zone", "eloquent"])
