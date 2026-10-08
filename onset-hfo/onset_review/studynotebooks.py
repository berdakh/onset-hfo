"""Each study page as a notebook that rebuilds it: Qt-free.

*Open as notebook* on a study page writes one of these and opens it on the
Analysis page, where every cell runs in the console beside the recording
that is open. The notebook reads the same committed tables through the same
loaders the page reads (``app/panels.py`` via `onset_review.studies`) and
draws the same figures with the same functions (`onset_review.studycharts`),
so what it shows is the page, taken apart -- a starting point for a reader's
own question about the study, not a second version of it.

The page's choices (band, metric, patient) are written into the first code
cell, where they can be changed. Nothing a notebook does is written back to
the committed tables.
"""

from __future__ import annotations

from pathlib import Path

from onset_review import cells

__all__ = ["notebook_script", "write_study_notebook"]

_SETUP = '''# %% The page's tables, through the loaders the page itself uses
import matplotlib.pyplot as plt
import pandas as pd

from onset_review import studies, studycharts

panels = studies._panels()
if panels is None:
    raise RuntimeError("The study tables are not on this machine: they ship with a "
                       "checkout of the repository and with the release bundle's site/.")
# The open recording, when this runs in the window's console; None otherwise.
session = globals().get("session")
'''

_DETECTORS = '''# %% The committed sweep: one row per band, detector and threshold
sweep = panels.sweep()
part = sweep[sweep["band"] == BAND]
part.pivot(index="threshold_sd", columns="detector", values=METRIC)

# %% The operating point each detector peaks at
best = panels.operating_points(CRITERION)
best

# %% The page's figure, drawn again
from onset_review import thisrecording

marks = thisrecording.marks(session, METRIC, BAND) if session is not None else []
figure, axis = plt.subplots(figsize=(9, 3.6))
studycharts.draw_sweep(figure, axis, sweep, BAND, METRIC, best, marks=marks)
figure.tight_layout()

# %% This window scored the way the study scored each patient (needs expert markings)
thisrecording.score_window(session) if session is not None else "no recording open"
'''

_OUTCOME = '''# %% One row per patient, as the Outcome page reads them
cohort = panels.cohort_overview()
cohort[["subject", "seizure_free", "candidates_resected_expert", "candidates_resected_rms"]]

# %% The page's figure, drawn again
from pathlib import Path

groups = pd.read_csv(Path(panels.STUDIES) / "outcome_groups_300s.csv")
groups = groups.query("scope == 'reviewed' and band == 'fast_ripple'")
figure = plt.figure(figsize=(9, 3.4))
studycharts.draw_outcome(figure, cohort, groups)
figure.tight_layout()

# %% The comparison behind the AUC, recomputed from the rows above
from onset_hfo.outcome import min_detectable_auc, rank_comparison

free = cohort.loc[cohort["seizure_free"].astype(bool), "candidates_resected_rms"]
recurrence = cohort.loc[~cohort["seizure_free"].astype(bool), "candidates_resected_rms"]
result = rank_comparison(free.to_numpy(float), recurrence.to_numpy(float))
print({k: result[k] for k in ("n_seizure_free", "n_recurrence", "auc", "auc_lo", "auc_hi",
                              "p_permutation")})
print("smallest detectable AUC:", min_detectable_auc(result["n_seizure_free"],
                                                     result["n_recurrence"]))
'''

_PATIENTS = '''# %% The cohort, one patient a row
cohort = panels.cohort_overview(band=BAND, scope=SCOPE)
cohort

# %% One patient
cohort[cohort["subject"] == SUBJECT].T if SUBJECT else "pick a SUBJECT above"
'''

_OTHER = '''# %% What the loaders offer, to start a question of your own
[name for name in dir(panels) if not name.startswith("_")]
'''


def _markdown(text: str) -> str:
    return "# %% [markdown]\n" + "\n".join(("# " + line) if line else "#"
                                          for line in text.strip().split("\n"))


def notebook_script(key: str, page_text: str = "", **choices) -> str:
    """The notebook for study page `key` as ``# %%`` cells."""
    from onset_review.studies import STUDIES

    titles = {k: label for k, label, _ in STUDIES}
    if key not in titles:
        raise KeyError(f"no study page {key!r}")
    intro = (f"# {titles[key]} — the study page as a notebook\n\n"
             "This notebook rebuilds the page from the committed tables, through the "
             "loaders the page itself uses. Run a cell with Ctrl+Enter. Change the "
             "choices in the first code cell and run again. Nothing here changes the "
             "published tables; what you compute is yours.")
    settings = ["# %% The page's choices: change them and run the cells below"]
    defaults = {"detectors": {"BAND": "ripple", "METRIC": "rank_rho",
                              "CRITERION": "rank_rho"},
                "patients": {"BAND": "fast_ripple", "SCOPE": "reviewed", "SUBJECT": ""}}
    names = {"BAND": "band", "METRIC": "metric", "CRITERION": "criterion",
             "SCOPE": "scope", "SUBJECT": "subject"}
    for name, default in defaults.get(key, {}).items():
        settings.append(f"{name} = {choices.get(names[name]) or default!r}")
    parts = [_markdown(intro), _SETUP.rstrip()]
    if len(settings) > 1:
        parts.append("\n".join(settings))
    parts.append({"detectors": _DETECTORS, "outcome": _OUTCOME,
                  "patients": _PATIENTS}.get(key, _OTHER).rstrip())
    if page_text.strip():
        parts.append(_markdown("## The page as published\n\n" + page_text.strip()))
    return "\n\n".join(parts) + "\n"


def write_study_notebook(key: str, folder: str | Path, page_text: str = "",
                         **choices) -> Path:
    """Write the notebook into `folder` as ``onset-<key>.ipynb``, never over
    an existing file: ``onset-<key>-2.ipynb`` and so on after the first."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"onset-{key}.ipynb"
    number = 1
    while path.exists():
        number += 1
        path = folder / f"onset-{key}-{number}.ipynb"
    return cells.write_notebook(notebook_script(key, page_text, **choices), path)
