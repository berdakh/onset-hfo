# Template: Does the ranking survive another threshold?
# Group: Events
# Mirrors: the threshold-sensitivity view on the Recording page
# About: The ranking detector re-run at lower and higher thresholds: which channel leads at each, and whether it still stands out from the rest.

import re

from onset_review import sensitivity

# %% Re-run at each multiple of the threshold (takes a few seconds)
sens = sensitivity.compute(session)
print(re.sub("<[^>]+>", "", sensitivity.describe(sens)))      # the view's sentence, as text

# %% Leader and tie set at each threshold
print(pd.DataFrame({"threshold (SD)": sens.thresholds, "leader": sens.leaders,
                    "stands out": sens.stands_out,
                    "tied": [", ".join(t) for t in sens.tied]}).to_string(index=False))

# %% Rates of the top channels at each threshold (rows: channels, columns: thresholds)
print(sens.rates.round(1).to_string())
if not sens.rates.empty:
    fig, ax = plt.subplots(figsize=(7, 4))
    for channel in sens.channels:
        ax.plot(sens.thresholds, sens.rates.loc[channel].to_numpy(), marker="o", label=channel)
    ax.set_xlabel("detector threshold (SD)")
    ax.set_ylabel("events / min")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
