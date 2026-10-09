# Template: Power spectrum of every channel
# Group: Signal
# Mirrors: the Spectrum panel
# About: Welch's spectrum of the preprocessed signal, the analysis band shaded and the mains lines marked, computed the way the panel does and by hand with MNE.

from mne.time_frequency import psd_array_welch

from onset_review import spectrum as spectrum_panel

# %% As the panel computes it
spec = spectrum_panel.compute(session)
print(spectrum_panel.summarise(spec).head(10).to_string(index=False))

# %% By hand, with MNE (µV²/Hz, 2 s segments)
data_uv = np.asarray(signal) * 1e6
power, freqs = psd_array_welch(data_uv, sfreq, fmin=1.0, fmax=min(sfreq / 2, 600.0),
                               n_per_seg=int(2 * sfreq), verbose=False)

# %% Plot the busiest channel and the median of all of them
busiest = findings["channel"].iloc[0]
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(freqs, np.median(power, axis=0), label="median of all channels", color="0.5")
ax.semilogy(freqs, power[channels.index(busiest)], label=busiest)
ax.axvspan(*request.band_hz, alpha=0.15, label="analysis band")
for line in np.arange(request.line_freq, freqs[-1], request.line_freq):
    ax.axvline(line, linestyle=":", linewidth=0.8, color="0.3")
ax.set_xlabel("Hz")
ax.set_ylabel("µV²/Hz")
ax.legend()
fig.tight_layout()
plt.show()
