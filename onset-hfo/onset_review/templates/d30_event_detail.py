# Template: One event in detail
# Group: Events
# Mirrors: the event detail view
# About: The strongest accepted event on the busiest channel: the raw trace, the band-passed one and a time-frequency map around it, made by hand with MNE, beside the panel's own reading of it.

from mne.filter import filter_data
from mne.time_frequency import tfr_array_morlet

from onset_review import detail

PAD_S = 0.25                     # seconds either side of the event

# %% Pick the event
busiest = findings["channel"].iloc[0]
chosen = [e for e in events if e.accepted and e.channel == busiest and e.detector == request.primary]
event = max(chosen, key=lambda e: e.score)
print(f"{busiest}: {event.start:.3f}-{event.stop:.3f} s, peak {event.peak_frequency_hz:.0f} Hz, "
      f"{event.peak_amplitude_uv:.0f} µV, {event.n_cycles:.1f} cycles")

# %% The panel's reading
snap = detail.snapshot(session, event)
print(detail.read_event(snap))

# %% By hand: cut, band-pass, time-frequency
i = channels.index(busiest)
start = max(0, int((event.start - PAD_S) * sfreq))
stop = min(signal.shape[1], int((event.stop + PAD_S) * sfreq))
piece = np.asarray(signal[i, start:stop]) * 1e6                   # µV
low, high = request.band_hz
band = filter_data(piece, sfreq, low, high, verbose=False)
freqs = np.arange(max(20.0, low / 2), min(high * 1.5, sfreq / 2 - 1), 5.0)
power = tfr_array_morlet(piece[None, None, :], sfreq, freqs, n_cycles=freqs / 10.0,
                         output="power", verbose=False)[0, 0]
t = (np.arange(start, stop) / sfreq) + t_offset

# %% Plot the three views
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
axes[0].plot(t, piece, linewidth=0.7)
axes[0].set_ylabel("raw (µV)")
axes[1].plot(t, band, linewidth=0.7)
axes[1].set_ylabel(f"{low:g}–{high:g} Hz (µV)")
axes[2].pcolormesh(t, freqs, 10 * np.log10(power + 1e-12), shading="auto")
axes[2].set_ylabel("Hz")
axes[2].set_xlabel("time in the recording (s)")
for ax in axes:
    ax.axvspan(event.start + t_offset, event.stop + t_offset, alpha=0.12)
fig.tight_layout()
plt.show()
