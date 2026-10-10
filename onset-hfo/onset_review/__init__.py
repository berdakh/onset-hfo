"""Onset Review: a desktop iEEG reviewer built on this project's pipeline.

`onset_hfo` is a library and a command line; `onset_review` is the thing a
clinician opens. It adds no analysis -- every rate, interval, candidate set and
caveat comes from `onset_hfo` -- and exists because a method a clinician cannot
sit with for an afternoon is a method nobody will tell you is wrong.

Three layers, deliberately separable:

* `session` and `trends` and `report` -- pure logic, no Qt. Everything the
  interface displays is computed, and tested, here.
* `panels`, `window`, `launcher` -- the Qt layer. It renders and navigates; it
  computes nothing.
* `app` -- the entry point. `onset-review` on the command line, or a desktop
  launcher.

Importing this package pulls in no Qt, so `--export` and the test suite work on
a machine with no display.
"""

from __future__ import annotations

__all__ = ["__version__", "__author__", "DEVELOPER", "HOMEPAGE", "credit"]

#: Tracks `onset-hfo`'s version: the reviewer is a face on that pipeline, and
#: two version numbers would only ever be a question about which one applies.
__version__ = "0.3.14"

#: Who made it: shown on the Home page, in Help → About, in `--version` and
#: in every exported report.
DEVELOPER = "Berdakh Abibullaev"
__author__ = DEVELOPER
HOMEPAGE = "https://berdakh.github.io/onset/"


def credit() -> str:
    """One line naming the software, its version and its developer."""
    return f"Onset Review {__version__} · developed by {DEVELOPER}"
