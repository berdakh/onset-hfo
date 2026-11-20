"""Where a high ripple rate is expected in a healthy brain.

Ripples are not only pathological. Hippocampus and the cortex around it
produce them during NREM sleep as part of memory consolidation; occipital
cortex and the sensorimotor strip produce them too. A contact in one of those
places with the highest ripple rate of the implant may be measuring the
patient's physiology, and no feature this software measures tells the two
apart (`docs/LIMITATIONS.md`).

So this module does one modest thing: given the name of the structure a
channel sits in -- from an atlas, or read off the electrode's name -- it says
whether that is one of those places. The result is a note beside a rate. It
never removes a channel, reweights a rate or changes a rank.
"""

from __future__ import annotations

import re

__all__ = ["expected_physiological", "physiological_note", "NOTE"]

#: Structure-name fragments, and what to call the region in a sentence.
_REGIONS = (
    (re.compile(r"hippocamp|parahippocamp|entorhinal|subicul", re.IGNORECASE),
     "mesial temporal (hippocampal formation)"),
    (re.compile(r"occipital|calcarine|cuneus|lingual|pericalcarine", re.IGNORECASE),
     "occipital"),
    (re.compile(r"precentral|postcentral|paracentral|rolandic|sensorimotor"
                r"|primary motor|central sulcus", re.IGNORECASE),
     "sensorimotor (peri-rolandic)"),
)

NOTE = ("Ripples occur physiologically in these regions, most of all in NREM "
        "sleep, and nothing measured here separates a physiological ripple "
        "from a pathological one. A high rate there is not by itself evidence "
        "of epileptogenic tissue.")


def expected_physiological(label: str) -> str:
    """The region's name if ripples are expected physiologically there, else ""."""
    text = str(label or "")
    for pattern, name in _REGIONS:
        if pattern.search(text):
            return name
    return ""


def physiological_note(channels, labels, how: str = "") -> str:
    """One paragraph naming the given channels that sit in such regions, or "".

    `channels` and `labels` run in step: the channels a reader will look at
    first (the highest rates) and the structure each is in. `how` says where
    the structure names came from -- "by electrode name", "by atlas label" --
    because a region read off a name is a claim about a convention, not about
    this patient's anatomy.
    """
    found = [(str(channel), expected_physiological(label))
             for channel, label in zip(channels, labels, strict=True)]
    found = [(channel, region) for channel, region in found if region]
    if not found:
        return ""
    listed = "; ".join(f"{channel} ({region})" for channel, region in found[:10])
    return (f"In a region where ripples are expected physiologically"
            f"{f', {how}' if how else ''}: {listed}"
            f"{'; …' if len(found) > 10 else ''}. {NOTE}")
