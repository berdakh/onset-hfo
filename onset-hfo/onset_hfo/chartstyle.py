"""Shared figure colours for core imaging and the desktop charts.

Keeping the validated series pair here lets library figures share the
application's colours without importing its window or chart modules.
"""

from __future__ import annotations

__all__ = ["SERIES"]

#: Categorical hues, in fixed order: the accent, then an orange. Under
#: simulated protan, deutan and tritan vision (Machado 2009, full severity) the
#: pair stays at least 111 CIE76 Delta-E apart; contrast on white 6.3:1 and 5.2:1.
SERIES = ("#4F46E5", "#C2410C")
