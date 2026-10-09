# Template: Event rate over time
# Group: Detection and ranking
# Mirrors: the trend under the trace
# About: Accepted events per minute in consecutive bins, for the whole window and for the busiest channel: is the activity steady, or one burst?

from onset_hfo.metrics import rate_timecourse

BIN_S = 5.0                      # bin width, seconds

# %% All channels together, and the busiest one alone
hfo = [e for e in events if e.detector == request.primary]
start, stop = span
whole = rate_timecourse(hfo, start - t_offset, stop - t_offset, bin_s=BIN_S)
busiest = findings["channel"].iloc[0]
alone = rate_timecourse(hfo, start - t_offset, stop - t_offset, bin_s=BIN_S,
                        channels=[busiest])

# %% Plot
fig, ax = plt.subplots(figsize=(8, 3))
ax.step(whole["t_start"] + t_offset, whole["rate_per_min"], where="post", label="all channels")
ax.step(alone["t_start"] + t_offset, alone["rate_per_min"], where="post", label=busiest)
ax.set_xlabel("time in the recording (s)")
ax.set_ylabel(f"events / min ({BIN_S:g} s bins)")
ax.legend()
fig.tight_layout()
plt.show()
print(f"{busiest}: {int(alone['n_events'].sum())} events; "
      f"busiest bin {alone['rate_per_min'].max():.0f}/min")
