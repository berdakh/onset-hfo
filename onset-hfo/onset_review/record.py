"""The patient record, as data: what the archive knows and what it does not.

Qt-free on purpose, like `session`, `trends`, `anatomy` and `report`. The panel
that draws this lives in `onset_review.patient`; what it is allowed to say
lives here, so the claims can be tested on a machine with no display -- and so
that the one rule that matters can be tested at all.

**It invents nothing.** There is no seizure semiology in `ds003498`, no imaging
report, no medication list, no neuropsychology, no prior surgery. A record that
filled those fields with plausible text would be a fabricated medical record,
and in software a clinician is being asked to trust, a fabricated record is the
single most dangerous thing it could contain. So `CHART_FIELDS` names them and
carries no content; the panel shows them empty, each saying where a site would
connect its own source.

Two things are deliberately absent even though the archive has them. Age, sex
and handedness are in `participants.tsv` and are not committed to this
repository: published beside pathology, surgical extent and outcome, three
demographic fields narrow a cohort of twenty considerably, and no analysis here
uses them. `data/outcome/README.md` records that decision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["patient_record", "CHART_FIELDS", "ILAE_CLASSES"]

#: ILAE surgical outcome classes, in the archive's own numbering.
ILAE_CLASSES = {
    1: "completely seizure-free, no auras",
    2: "auras only, no other seizures",
    3: "1–3 seizure days a year",
    4: "4 seizure days a year to 50% reduction",
    5: "less than 50% reduction up to 100% increase",
    6: "more than 100% increase in seizure days",
}

#: The clinical-chart fields a hospital system would supply, in the order a
#: neurophysiologist would read them. Present so a reviewer can see where their
#: own data lands; **never** populated with anything invented.
CHART_FIELDS = [
    ("Seizure semiology", "what the seizures look like, and the aura"),
    ("Age at onset / duration", "how long the epilepsy has been going on"),
    ("MRI", "the radiologist's report, and whether a lesion was found"),
    ("PET / SPECT / MEG", "other localising investigations"),
    ("Scalp EEG", "interictal discharges and ictal onset on scalp"),
    ("Neuropsychology", "baseline function, and what is at risk"),
    ("Medication", "current anti-seizure medication and past trials"),
    ("Prior surgery", "earlier resections or neurostimulation"),
]


def patient_record(subject: str, data_dir: Path | None = None) -> dict:
    """Everything the committed tables hold about one patient.

    Reads the extracts in `data/outcome/`, which exist precisely so a number
    can be checked from a fresh clone with no download. Missing files are not
    an error: the panel says the record is unavailable, which is true and is
    more useful than a traceback.
    """
    root = Path(data_dir) if data_dir is not None else _default_dir()
    record: dict = {"subject": subject, "available": False}
    participants = _read(root / "participants.csv")
    if participants is None or "subject" not in participants:
        return record
    rows = participants[participants["subject"] == subject]
    if rows.empty:
        return record

    row = rows.iloc[0]
    record.update(available=True, epilepsy=str(row.get("epilepsy", "") or ""),
                  ilae=_int(row.get("ilae")), nights=_int(row.get("nights")),
                  outcome=str(row.get("outcome", "") or ""),
                  months=_int(row.get("months_follow_up")),
                  lesion=_int(row.get("lesion")))

    recordings = _read(root / "recordings.csv")
    if recordings is not None and "subject" in recordings:
        runs = recordings[recordings["subject"] == subject]
        if not runs.empty:
            run = runs.iloc[0]
            missing = str(run.get("missing", "") or "")
            record.update(
                n_channels=_int(run.get("n_channels")),
                n_reviewed=_int(run.get("n_reviewed")),
                n_resected=_int(run.get("n_resected_channels")),
                n_partial=_int(run.get("n_partial_channels")),
                n_eloquent=_int(run.get("n_eloquent_channels")),
                n_expert_events=_int(run.get("n_expert_events")),
                rz_coverage=_float(run.get("rz_coverage")),
                missing=[c.strip() for c in missing.split(",") if c.strip()
                         and missing.lower() != "nan"],
            )
    return record


def _default_dir() -> Path:
    """`data/outcome/` relative to the installed package."""
    return Path(__file__).resolve().parent.parent / "data" / "outcome"


def _read(path: Path):
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number
