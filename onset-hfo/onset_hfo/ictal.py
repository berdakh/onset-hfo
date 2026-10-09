"""Where a seizure starts: the Epileptogenicity Index, per contact. No Qt.

At a seizure's onset the contacts that start it change first and change
most: their activity shifts from slow rhythms to fast ones ("low-voltage
fast activity"). The Epileptogenicity Index (Bartolomei, Chauvel & Wendling,
Brain 2008) puts a number on both at once:

1. **Energy ratio**, per contact, in sliding windows: the energy in the fast
   bands (beta and gamma, 12.4-97 Hz) over the energy in the slow ones (theta
   and alpha, 3.5-12.4 Hz). Here it is divided by the contact's own median in
   a baseline before the seizure, so 1 is "as usual".
2. **When it changed**: a Page-Hinkley cumulative-sum test on that ratio
   raises an alarm when it has risen and stayed risen; the change is dated to
   the point where the cumulative sum last turned upward, before the alarm.
3. **The index**: the ratio summed over `h_s` seconds from the change, divided
   by how long after the *first* contact's change it came (plus `tau_s`), then
   scaled so the highest contact is 1. A contact that changes early and
   strongly scores near 1; late or weakly, near 0; never, 0.

Across several seizures, `combine` keeps each contact's median index and how
many seizures it scored at least `EI_CUTOFF` in -- the onset zone of one
seizure is not necessarily the onset zone, and a contact that leads in every
seizure is a different finding from one that led once.

The constants are the published method's where it gives them (the bands,
tau, h); the Page-Hinkley bias and threshold are stated, used everywhere,
and reported with every result. This is a research measure. It needs the
seizure's electrographic onset marked by a person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = ["IctalSettings", "energy_ratio", "page_hinkley", "epileptogenicity",
           "combine", "EI_CUTOFF", "EI_COLUMNS"]

#: An index at or above this is taken as "involved early and strongly"
#: (the threshold the method's authors used to delineate the zone).
EI_CUTOFF = 0.3
EI_COLUMNS = ("channel", "ei", "detected", "change_s", "alarm_s", "er_after", "rank")


@dataclass(frozen=True)
class IctalSettings:
    slow_band: tuple = (3.5, 12.4)
    fast_band: tuple = (12.4, 97.0)
    window_s: float = 1.0
    step_s: float = 0.25
    #: Baseline: from `pre_s` before the marked onset to `baseline_gap_s` before it.
    pre_s: float = 30.0
    baseline_gap_s: float = 5.0
    #: Where the change is looked for, around the marked onset.
    search_before_s: float = 10.0
    post_s: float = 30.0
    #: Page-Hinkley bias (in baseline-normalised ratio units) and threshold.
    bias: float = 0.5
    threshold: float = 15.0
    tau_s: float = 1.0
    h_s: float = 5.0

    def describe(self) -> str:
        return (f"energy ratio {self.fast_band[0]:g}–{self.fast_band[1]:g} Hz over "
                f"{self.slow_band[0]:g}–{self.slow_band[1]:g} Hz in {self.window_s:g} s windows "
                f"every {self.step_s:g} s, against a baseline from {self.pre_s:g} to "
                f"{self.baseline_gap_s:g} s before the marked onset; Page–Hinkley bias "
                f"{self.bias:g}, threshold {self.threshold:g}; change looked for from "
                f"{self.search_before_s:g} s before to {self.post_s:g} s after the onset; "
                f"index over {self.h_s:g} s, tau {self.tau_s:g} s")

    def to_json(self) -> dict:
        return asdict(self)


def energy_ratio(data: np.ndarray, sfreq: float, settings: IctalSettings = IctalSettings()
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Fast-over-slow energy per contact per window. `data` is contacts x
    samples. Returns (window centres in seconds from the first sample,
    ratios contacts x windows)."""
    data = np.asarray(data, dtype=float)
    width = max(8, int(round(settings.window_s * sfreq)))
    step = max(1, int(round(settings.step_s * sfreq)))
    if data.shape[1] < width:
        raise ValueError("the signal is shorter than one window")
    starts = np.arange(0, data.shape[1] - width + 1, step)
    taper = np.hanning(width)
    freqs = np.fft.rfftfreq(width, 1.0 / sfreq)
    slow = (freqs >= settings.slow_band[0]) & (freqs < settings.slow_band[1])
    fast = (freqs >= settings.fast_band[0]) & (freqs < min(settings.fast_band[1],
                                                           sfreq / 2))
    if not slow.any() or not fast.any():
        raise ValueError(f"at {sfreq:g} Hz the bands do not fit in a {settings.window_s:g} s "
                         "window")
    ratios = np.empty((data.shape[0], len(starts)))
    for i, start in enumerate(starts):
        block = data[:, start:start + width]
        block = (block - block.mean(axis=1, keepdims=True)) * taper
        power = np.abs(np.fft.rfft(block, axis=1)) ** 2
        ratios[:, i] = power[:, fast].sum(axis=1) / np.maximum(power[:, slow].sum(axis=1),
                                                                1e-30)
    centres = (starts + width / 2) / sfreq
    return centres, ratios


def page_hinkley(x: np.ndarray, bias: float, threshold: float) -> tuple[int | None, int | None]:
    """Page-Hinkley test for an upward change in `x`. Returns (alarm index,
    change index) or (None, None). U_n = sum_k<=n (x_k - mean(x_<=k) - bias);
    the alarm is the first n where U_n - min U > threshold; the change is the
    index of that minimum."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None, None
    running_mean = np.cumsum(x) / np.arange(1, len(x) + 1)
    u = np.cumsum(x - running_mean - bias)
    lowest = np.minimum.accumulate(u)
    over = np.flatnonzero(u - lowest > threshold)
    if not len(over):
        return None, None
    alarm = int(over[0])
    change = int(np.argmin(u[:alarm + 1]))
    return alarm, change


def epileptogenicity(data: np.ndarray, ch_names: list[str], sfreq: float, onset_s: float,
                     settings: IctalSettings = IctalSettings(), start_s: float = 0.0
                     ) -> pd.DataFrame:
    """The index of every contact for one seizure. `data` (contacts x samples,
    any unit) starts at `start_s`; `onset_s` is the marked electrographic
    onset in the same time base. Times in the result are seconds from the
    marked onset (negative: before it)."""
    centres, ratios = energy_ratio(data, sfreq, settings)
    times = centres + float(start_s) - float(onset_s)
    base = (times >= -settings.pre_s) & (times <= -settings.baseline_gap_s)
    # At least half the baseline stretch must be in the data: a few seconds of
    # "usual" is not a baseline.
    if base.sum() < 3 or base.sum() * settings.step_s < 0.5 * (settings.pre_s
                                                               - settings.baseline_gap_s):
        raise ValueError(f"need at least {settings.pre_s:g} s before the onset for a baseline")
    baseline = np.median(ratios[:, base], axis=1, keepdims=True)
    normalised = ratios / np.maximum(baseline, 1e-30)
    search = (times >= -settings.search_before_s) & (times <= settings.post_s)
    idx = np.flatnonzero(search)
    changes = np.full(len(ch_names), np.nan)
    alarms = np.full(len(ch_names), np.nan)
    for c in range(len(ch_names)):
        alarm, change = page_hinkley(normalised[c, idx], settings.bias, settings.threshold)
        if alarm is not None:
            alarms[c] = times[idx[alarm]]
            changes[c] = times[idx[change]]
    detected = np.isfinite(changes)
    first = np.nanmin(changes) if detected.any() else np.nan
    h = max(1, int(round(settings.h_s / settings.step_s)))
    raw_ei = np.zeros(len(ch_names))
    after = np.full(len(ch_names), np.nan)
    for c in np.flatnonzero(detected):
        k = int(np.searchsorted(times, changes[c]))
        energy = normalised[c, k:k + h].sum()
        after[c] = float(normalised[c, k:k + h].mean()) if k < normalised.shape[1] else np.nan
        raw_ei[c] = energy / (changes[c] - first + settings.tau_s)
    top = raw_ei.max()
    ei = raw_ei / top if top > 0 else raw_ei
    frame = pd.DataFrame({"channel": list(ch_names), "ei": ei, "detected": detected,
                          "change_s": changes, "alarm_s": alarms, "er_after": after})
    frame["rank"] = frame["ei"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values(["rank", "channel"]).reset_index(drop=True)[list(EI_COLUMNS)]


def combine(seizures: dict[str, pd.DataFrame], cutoff: float = EI_CUTOFF) -> pd.DataFrame:
    """Several seizures' indices, per contact: median index, how many seizures
    it reached `cutoff` in, its median delay from the marked onset, and in
    how many seizures it was analysed."""
    rows = []
    channels: list[str] = []
    for frame in seizures.values():
        for name in frame["channel"]:
            if name not in channels:
                channels.append(name)
    for name in channels:
        values, delays = [], []
        for frame in seizures.values():
            hit = frame[frame["channel"] == name]
            if len(hit):
                values.append(float(hit["ei"].iloc[0]))
                if bool(hit["detected"].iloc[0]):
                    delays.append(float(hit["change_s"].iloc[0]))
        rows.append({"channel": name, "median_ei": float(np.median(values)) if values else 0.0,
                     "seizures_high": int(sum(v >= cutoff for v in values)),
                     "seizures": len(values),
                     "median_change_s": float(np.median(delays)) if delays else np.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = out["median_ei"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["rank", "channel"]).reset_index(drop=True)
