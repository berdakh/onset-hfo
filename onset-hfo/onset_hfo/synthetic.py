"""A synthetic iEEG recording with known, implanted events.

Why a simulator is part of a data-driven prototype:

* **The public archive has no HFO labels.** Nobody has marked every ripple in
  ds003029 by hand, so on real data we can report rates and agreement but not
  precision and recall. On synthetic data we know exactly where every event
  is, so the detectors can be *measured* rather than admired.
* **It runs offline.** Tests, continuous integration and anyone behind a
  firewall get a full end-to-end run with no download.
* **It contains the traps on purpose.** Sharp transients that ring through an
  80-250 Hz filter, interictal spikes with and without ripples on top, line
  noise and its harmonics. A detector that scores well here has had to earn it.

What is simulated, per contact:

* 1/f ("pink") background noise, the dominant feature of real iEEG;
* mains noise at ``line_freq`` and its harmonics;
* **ripples**: Gaussian-windowed sinusoids, 80-200 Hz, 30-60 ms;
* **interictal discharges**: sharp biphasic waves, some with a ripple riding
  on them (as in real epileptic tissue);
* **artifacts**: large fast steps with broadband energy (electrode pops and
  movement transients reach hundreds of microvolts in real recordings) -- the
  classic false-HFO generator;
* a few "hot" contacts with a much higher event rate, standing in for the
  tissue an epileptologist would care about, and an optional "seizure" window
  in which their rate rises further.

The ground truth is returned as a table; :mod:`onset_hfo.evaluate` matches
detector output against it.
"""

from __future__ import annotations

import warnings

import mne
import numpy as np
import pandas as pd

from onset_hfo.datasets import Recording
from onset_hfo.detectors.base import bandpass

__all__ = ["make_synthetic_recording", "SyntheticSpec"]


class SyntheticSpec:
    """Default generation parameters, exposed so a notebook can vary them."""

    n_leads = 3                 #: electrodes ("SA", "SB", "SC")
    contacts_per_lead = 6       #: contacts per electrode
    sfreq = 2000.0              #: Hz -- high enough for ripples and fast ripples
    duration_s = 120.0
    background_uv = 30.0        #: RMS of the pink-noise background
    line_freq = 60.0
    line_uv = 8.0
    hot_leads = 1               #: leads whose middle contacts are "epileptic"
    hfo_rate_hot = 40.0         #: ripples per minute on a hot contact
    hfo_rate_cold = 2.0         #: ripples per minute elsewhere
    spike_rate_hot = 30.0       #: discharges per minute on a hot contact
    spike_rate_cold = 2.0
    artifact_rate = 4.0         #: artifacts per minute, on every contact
    ripple_snr = 7.0            #: peak amplitude / band-limited background RMS
    seizure_fraction = (0.6, 0.8)   #: portion of the recording with elevated rates
    seizure_gain = 4.0


def _pink_noise(n: int, sfreq: float, rng: np.random.Generator) -> np.ndarray:
    """Noise with a 1/f amplitude spectrum, normalised to unit RMS."""
    spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    x = np.fft.irfft(spectrum * scale, n=n)
    return x / (np.std(x) or 1.0)


def _ripple(sfreq: float, freq: float, duration: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(round(duration * sfreq))) / sfreq
    envelope = np.exp(-0.5 * ((t - duration / 2) / (duration / 6)) ** 2)
    return amplitude * envelope * np.sin(2 * np.pi * freq * (t - duration / 2))


def _spike(sfreq: float, duration: float, amplitude: float) -> np.ndarray:
    """A sharp biphasic discharge: cusp-like negative peak, slow positive recovery.

    The apex is a cusp (``exp(-|t|/tau)``), not a smooth Gaussian, because that
    is what real interictal spikes look like -- and because a cusp has energy
    at every frequency. Run it through an 80-250 Hz filter and it *rings*: the
    single most common source of false "ripples" in the HFO literature. The
    simulator has to contain the trap, or the artifact-rejection stage would
    be validated against a problem that never occurs.
    """
    n = int(round(duration * sfreq))
    t = np.linspace(-3, 3, n)
    sharp = -np.exp(-np.abs(t) / 0.35)
    slow = 0.45 * np.exp(-0.5 * ((t - 1.8) / 1.2) ** 2)
    wave = sharp + slow
    return amplitude * wave / (np.max(np.abs(wave)) or 1.0)


def _artifact(sfreq: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """A fast step with an exponential recovery: broadband, no oscillation.

    This is what makes a naive band-pass detector report ripples that are not
    there, and what the validation stage in :mod:`onset_hfo.validate` exists
    to remove.
    """
    n = int(round(0.06 * sfreq))
    t = np.arange(n) / sfreq
    step = np.concatenate([np.zeros(2), np.ones(n - 2)])
    decay = np.exp(-t / 0.01)
    return amplitude * np.sign(rng.normal()) * step * decay


def _poisson_times(rate_per_min: float, duration: float, rng: np.random.Generator,
                   gain_window: tuple[float, float] | None = None,
                   gain: float = 1.0, margin: float = 0.2) -> np.ndarray:
    """Event times from a Poisson process, optionally denser inside a window."""
    lam = rate_per_min / 60.0
    base = rng.uniform(margin, duration - margin,
                       size=rng.poisson(lam * duration))
    if gain_window and gain > 1:
        lo, hi = gain_window
        extra = rng.uniform(lo, hi, size=rng.poisson(lam * (gain - 1) * (hi - lo)))
        base = np.concatenate([base, extra])
    return np.sort(base)


def make_synthetic_recording(spec: SyntheticSpec | None = None, seed: int = 7,
                             verbose: bool = True, **overrides) -> Recording:
    """Generate a labelled synthetic recording.

    Parameters
    ----------
    spec:
        Generation parameters; defaults to :class:`SyntheticSpec`.
    seed:
        Random seed. The same seed always produces the same recording, which
        is what makes the tests meaningful.
    overrides:
        Any :class:`SyntheticSpec` attribute, e.g. ``duration_s=30``.

    Returns
    -------
    Recording
        With ``ground_truth`` set: one row per implanted event, with columns
        ``kind`` (ripple / spike / artifact), ``contact``, ``start``, ``stop``,
        ``frequency_hz`` and ``amplitude_uv``.
    """
    spec = spec or SyntheticSpec()
    for key, value in overrides.items():
        if not hasattr(spec, key):
            raise TypeError(f"Unknown synthetic parameter: {key}")
        setattr(spec, key, value)

    rng = np.random.default_rng(seed)
    sf = float(spec.sfreq)
    n = int(round(spec.duration_s * sf))
    leads = [f"S{chr(ord('A') + i)}" for i in range(spec.n_leads)]
    contacts = [f"{lead}{i}" for lead in leads for i in range(1, spec.contacts_per_lead + 1)]
    hot = {f"{leads[0]}{i}" for i in range(2, 2 + max(1, spec.contacts_per_lead // 3))} \
        if spec.hot_leads else set()
    seizure = (spec.duration_s * spec.seizure_fraction[0],
               spec.duration_s * spec.seizure_fraction[1])

    data = np.zeros((len(contacts), n))
    truth: list[dict] = []
    t_axis = np.arange(n) / sf

    for row, contact in enumerate(contacts):
        signal = spec.background_uv * _pink_noise(n, sf, rng)
        for harmonic in (1, 2, 3):
            signal += (spec.line_uv / harmonic) * np.sin(
                2 * np.pi * spec.line_freq * harmonic * t_axis + rng.uniform(0, 2 * np.pi))
        # Band-limited background level sets the amplitude of implanted
        # ripples, so that "SNR 6" means the same thing on every channel: the
        # ripple's peak is 6x the RMS the ripple band already carries.
        band_rms = float(np.std(bandpass(signal[None, :], sf, (80.0, 250.0)))) or 1.0
        is_hot = contact in hot

        for t0 in _poisson_times(spec.hfo_rate_hot if is_hot else spec.hfo_rate_cold,
                                 spec.duration_s, rng, seizure,
                                 spec.seizure_gain if is_hot else 1.0):
            freq = rng.uniform(85, 200)
            duration = rng.uniform(0.03, 0.06)
            amplitude = spec.ripple_snr * band_rms * rng.uniform(0.8, 1.4)
            wave = _ripple(sf, freq, duration, amplitude)
            i0 = int(round(t0 * sf))
            signal[i0:i0 + wave.size] += wave[:max(0, min(wave.size, n - i0))]
            truth.append({"kind": "ripple", "contact": contact, "start": t0,
                          "stop": t0 + duration, "frequency_hz": freq,
                          "amplitude_uv": float(np.max(np.abs(wave)))})

        for t0 in _poisson_times(spec.spike_rate_hot if is_hot else spec.spike_rate_cold,
                                 spec.duration_s, rng, seizure,
                                 spec.seizure_gain if is_hot else 1.0):
            duration = rng.uniform(0.05, 0.09)
            amplitude = rng.uniform(8, 16) * spec.background_uv
            wave = _spike(sf, duration, amplitude)
            i0 = int(round(t0 * sf))
            signal[i0:i0 + wave.size] += wave[:max(0, min(wave.size, n - i0))]
            truth.append({"kind": "spike", "contact": contact, "start": t0,
                          "stop": t0 + duration, "frequency_hz": float("nan"),
                          "amplitude_uv": amplitude})
            # In epileptic tissue a fraction of discharges carry a ripple.
            if is_hot and rng.random() < 0.4:
                freq = rng.uniform(90, 180)
                rip_dur = rng.uniform(0.03, 0.05)
                rip = _ripple(sf, freq, rip_dur, spec.ripple_snr * band_rms * 1.2)
                j0 = i0 + int(round(0.2 * duration * sf))
                signal[j0:j0 + rip.size] += rip[:max(0, min(rip.size, n - j0))]
                truth.append({"kind": "ripple", "contact": contact,
                              "start": j0 / sf, "stop": j0 / sf + rip_dur,
                              "frequency_hz": freq,
                              "amplitude_uv": float(np.max(np.abs(rip)))})

        for t0 in _poisson_times(spec.artifact_rate, spec.duration_s, rng):
            wave = _artifact(sf, rng.uniform(20, 60) * spec.background_uv, rng)
            i0 = int(round(t0 * sf))
            signal[i0:i0 + wave.size] += wave[:max(0, min(wave.size, n - i0))]
            truth.append({"kind": "artifact", "contact": contact, "start": t0,
                          "stop": t0 + wave.size / sf, "frequency_hz": float("nan"),
                          "amplitude_uv": float(np.max(np.abs(wave)))})

        data[row] = signal

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = mne.create_info(contacts, sf, ch_types="seeg", verbose="ERROR")
        raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # microvolts -> volts

    ground_truth = pd.DataFrame(truth).sort_values(["start", "contact"]).reset_index(drop=True)
    rec = Recording(
        raw=raw, source="synthetic", subject="sim-01", task="simulated", run="01",
        line_freq=spec.line_freq, dataset_id="synthetic",
        t_offset=0.0, seizure=seizure, bads=[], events=None, channels=None,
        marked_contacts=sorted(hot), ground_truth=ground_truth,
        citation="Synthetic recording generated by onset_hfo.synthetic (no patient data).",
        notes=[
            f"synthetic recording, seed={seed}: {len(contacts)} contacts, "
            f"{spec.duration_s:g} s @ {sf:g} Hz",
            f"implanted: {int((ground_truth['kind'] == 'ripple').sum())} ripples, "
            f"{int((ground_truth['kind'] == 'spike').sum())} discharges, "
            f"{int((ground_truth['kind'] == 'artifact').sum())} artifacts",
            f"'hot' contacts with elevated rates: {', '.join(sorted(hot)) or 'none'}",
            "ground truth is known exactly; use onset_hfo.evaluate for precision/recall",
        ],
    )
    if verbose:
        print(f"[onset-hfo] synthetic recording: {len(contacts)} contacts, {spec.duration_s:g} s "
              f"@ {sf:g} Hz, {len(ground_truth)} implanted events "
              f"(hot contacts: {', '.join(sorted(hot))})")
    return rec


#: The 18 scalp electrodes of the double banana, one per synthetic contact.
_SCALP_NAMES = ("Fp1", "F7", "T7", "P7", "O1", "F3", "C3", "P3", "Fp2", "F4", "C4",
                "P4", "O2", "F8", "T8", "P8", "Fz", "Cz")


def as_modality(rec: Recording, modality: str) -> Recording:
    """The same synthetic recording dressed as scalp EEG or MEG.

    Its signal and implanted events are unchanged; only the channel names,
    the channel types and, for MEG, the unit change: 10-20 electrode names
    typed ``eeg`` for ``"eeg"``, Neuromag-style names typed ``grad`` (T/m,
    scaled so 50 µV becomes 500 fT/cm) for ``"meg_grad"`` and ``mag`` (T,
    50 µV becomes 500 fT) for ``"meg_mag"``. It exists so the scalp and MEG
    paths can be tested end to end without a download; it says nothing about
    how real scalp or MEG HFOs look.
    """
    raw = rec.raw.copy()
    old = list(raw.ch_names)
    if modality == "eeg":
        if len(old) > len(_SCALP_NAMES):
            raise ValueError(f"only {len(_SCALP_NAMES)} scalp names to give")
        new, kind, factor = list(_SCALP_NAMES[:len(old)]), "eeg", 1.0
    elif modality in ("meg_grad", "meg_mag"):
        kind = "grad" if modality == "meg_grad" else "mag"
        new = [f"MEG{1000 + 10 * i + (2 if kind == 'grad' else 1):04d}"
               for i in range(len(old))]
        # 1 µV -> 10 fT/cm (1e-12 T/m) or 10 fT (1e-14 T).
        factor = 1e-6 if kind == "grad" else 1e-8
    else:
        raise ValueError(f"modality must be eeg, meg_grad or meg_mag, not {modality!r}")
    mapping = dict(zip(old, new, strict=True))
    raw.rename_channels(mapping)
    raw.set_channel_types({name: kind for name in new}, on_unit_change="ignore",
                          verbose="ERROR")
    if factor != 1.0:
        raw._data *= factor
    truth = rec.ground_truth.copy() if rec.ground_truth is not None else None
    if truth is not None and "contact" in truth:
        truth["contact"] = truth["contact"].map(lambda c: mapping.get(c, c))
    return Recording(
        raw=raw, source=f"{rec.source}:{modality}", subject=rec.subject, task=rec.task,
        run=rec.run, t_offset=rec.t_offset, seizure=rec.seizure, bads=[],
        marked_contacts=[mapping.get(c, c) for c in rec.marked_contacts],
        ground_truth=truth, line_freq=rec.line_freq,
        notes=list(rec.notes) + [f"synthetic recording dressed as {modality}"])
