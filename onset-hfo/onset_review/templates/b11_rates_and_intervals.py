# Template: Rates per channel, with their intervals
# Group: Detection and ranking
# Mirrors: the Recording page's ranking
# About: The ranking as a chart: each channel's rate with its 95% Poisson interval, and the interval recomputed by hand from the count and the minutes analysed.

from onset_hfo.metrics import poisson_ci

# %% The table behind the ranking
table = findings[["channel", "n_events", "duration_s", "rate_per_min",
                  "rate_ci_low", "rate_ci_high"]].copy()
print(table.head(15).to_string(index=False))

# %% The interval by hand: an exact Poisson interval on the count, per minute analysed
row = table.iloc[0]
low, high = poisson_ci(int(row["n_events"]), float(row["duration_s"]) / 60.0)
print(f"{row['channel']}: {row['rate_per_min']:.1f}/min, interval {low:.1f}-{high:.1f} "
      f"(the window says {row['rate_ci_low']:.1f}-{row['rate_ci_high']:.1f})")

# %% Plot it: overlapping intervals are channels the data cannot tell apart
top = table.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 0.3 * len(top) + 1))
ax.errorbar(top["rate_per_min"], range(len(top)),
            xerr=[top["rate_per_min"] - top["rate_ci_low"],
                  top["rate_ci_high"] - top["rate_per_min"]], fmt="o", capsize=3)
ax.set_yticks(range(len(top)), top["channel"])
ax.set_xlabel("events per minute (95% interval)")
ax.set_title(f"{request.subject} · {request.band_label()}")
fig.tight_layout()
plt.show()
