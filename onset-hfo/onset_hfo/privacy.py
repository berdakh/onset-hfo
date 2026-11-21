"""What must not travel with an analysis: a patient's name in a file name.

A hospital exports a recording as `SMITH_JOHN_19800101.edf`. Unless somebody
types a study code over it, that stem becomes the subject label, and the label
goes everywhere a label goes: the report's title, exported file names, the
application log, a problem-report bundle, and the prompt sent to a language
model. None of those is a place for a name and a date of birth.

This module does not de-identify recordings (`onset_hfo.case.deid` checks a
case's text before a report). It answers three narrow questions the single-
recording path needs answered before anything is labelled or sent:

* does this text look like it identifies a person (:func:`looks_identifying`);
* what to call the recording instead (:func:`study_code`, :func:`safe_label`,
  :func:`shown_name`);
* does this address leave the machine (:func:`is_loopback`).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["looks_identifying", "study_code", "safe_label", "shown_name",
           "is_loopback", "host_of"]

#: Words a recording's file name is made of that are not anybody's name.
_ORDINARY = frozenset("""
sub ses task run acq rec desc space ieeg eeg seeg ecog meg raw edf bdf fif vhdr
study case patient pt subject imported export exported recording record data
file test demo sample clinical rest sleep wake awake night day ictal interictal
seizure baseline pre post left right implant session part segment clip long
short full final new old copy anon anonymous deid montage bipolar monopolar
""".split())

_DATE = re.compile(r"(?<!\d)(?:\d{1,2}[-_.]\d{1,2}[-_.](?:19|20)?\d{2}"
                   r"|(?:19|20)\d{2}[-_.]?\d{2}[-_.]?\d{2}"
                   r"|\d{2}[-_.]?\d{2}[-_.]?(?:19|20)\d{2})(?!\d)")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{6,}(?!\d)")
_WORD = re.compile(r"[A-Za-z]{3,}")


def looks_identifying(text: str) -> str:
    """Why `text` looks like it names a person, or "" when it does not.

    A heuristic on purpose, and a cautious one: a date, a number long enough
    to be a hospital number, or two words that are not the ordinary
    vocabulary of a recording's file name (`sub`, `task`, `rest`, `ieeg`...).
    It errs toward saying yes -- the cost of a false alarm is a study code
    where a harmless stem would have done.
    """
    text = str(text or "")
    if _DATE.search(text):
        return "it contains what looks like a date"
    if _LONG_NUMBER.search(text):
        return "it contains a number long enough to be a hospital number"
    words = [w for w in _WORD.findall(text) if w.lower() not in _ORDINARY]
    if len(words) >= 2:
        return "it contains what may be a name"
    return ""


def study_code(path) -> str:
    """A label for a recording that says nothing about whose it is.

    Derived from the file's name and size, so the same file gets the same
    code every time it is opened and a verdict file can be found again. Six
    hexadecimal digits is enough to tell a ward's recordings apart and is
    not a hash anybody is meant to reverse; the name is not recoverable from
    the report it is printed in.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    digest = hashlib.sha256(f"{path.name}|{size}".encode()).hexdigest()
    return f"study-{digest[:6]}"


def safe_label(path) -> str:
    """The file's stem when it is harmless, else its :func:`study_code`."""
    stem = Path(path).name.split(".")[0]
    return study_code(path) if looks_identifying(stem) or not stem else stem[:40]


def shown_name(path) -> str:
    """The file's name as a report, a note or a prompt may print it."""
    path = Path(path)
    if not looks_identifying(path.name.split(".")[0]):
        return path.name
    return f"{study_code(path)}{''.join(path.suffixes[-2:]) or path.suffix}"


def host_of(url: str) -> str:
    try:
        return (urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return ""


def is_loopback(url: str) -> bool:
    """Whether an address stays on this machine. An address that cannot be
    read does not count as local."""
    host = host_of(url)
    return host in ("localhost", "::1") or host.startswith("127.") or host == "[::1]"
