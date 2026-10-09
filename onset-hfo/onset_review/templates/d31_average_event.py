# Template: The average event on a channel
# Group: Events
# Mirrors: the average-event view
# About: Every accepted event on the busiest channel aligned on its peak and averaged: a consistent oscillation averages into one, a mixture of artefacts into a smear.

import re

from onset_review import average

# %% Average, as the view does
busiest = findings["channel"].iloc[0]
avg = average.compute(session, busiest)
print(re.sub("<[^>]+>", "", average.describe(avg)))           # the view's sentence, as text

# %% Plot mean ± SD, wideband and band-passed
if avg.n:
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ax, mean, sd, label in ((axes[0], avg.mean_wideband, avg.sd_wideband, "wideband"),
                                (axes[1], avg.mean_band, avg.sd_band, "band-passed")):
        ax.plot(avg.times * 1e3, mean)
        ax.fill_between(avg.times * 1e3, mean - sd, mean + sd, alpha=0.25)
        ax.set_ylabel(f"{label} (µV)")
    axes[1].set_xlabel("ms from the peak")
    axes[0].set_title(f"{busiest}: {avg.n} of {avg.n_available} events")
    fig.tight_layout()
    plt.show()
