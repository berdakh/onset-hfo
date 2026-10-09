# Template: Events inside and outside the resection
# Group: Anatomy
# Mirrors: the Contacts page's resection rings and the Outcome study
# About: Each channel labelled resected, partial (one contact removed) or spared from the surgeon's resected zone, and the share of events in each -- the outcome study's measure, for this window.

from onset_hfo.clinical import classify_channels

# %% Needs the resected zone (the archive's clinical sheet)
if resection is None:
    print("No resected zone is known for this recording, so there is nothing to compare. "
          "Public ds003498 windows have one; an imported file does not.")
else:
    zones = classify_channels(list(findings["channel"]), resection).set_index("channel")["zone"]
    counts = findings.set_index("channel")["n_events"]
    print(resection.coverage(list(channels)))
    share = counts.groupby(zones).sum() / max(counts.sum(), 1)
    print(share.round(3).to_string())
    top = counts.idxmax()
    print(f"busiest channel {top} is {zones[top]}")
