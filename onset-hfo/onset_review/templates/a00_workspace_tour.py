# Template: What is in the workspace
# Group: Start here
# Mirrors: the Workspace pane
# About: Every name the console starts with, what it holds and its shape: the place to begin before writing anything.

# %% The recording on screen
print(request.label())                     # subject, window, band, detectors
print(f"{len(channels)} channels at {sfreq:g} Hz, montage: {montage}")
print(f"signal: {signal.shape} (channels x samples, volts, read-only)")
print(f"analysed span: {span[0]:g} to {span[1]:g} s of the original recording")

# %% What the window found
print(f"{len(events)} events from {sorted({e.detector for e in events})}, "
      f"{sum(e.accepted for e in events)} accepted")
print(findings[["channel", "n_events", "rate_per_min", "rate_ci_low", "rate_ci_high"]]
      .head(10).to_string(index=False))
print(leader["statement"] if leader.get("available") else "no ranking")
print("tied with the busiest:", candidates)

# %% Quality, anatomy and your read
if quality is not None:
    print(f"{int((~quality['good']).sum())} of {len(quality)} channels set aside")
print("electrode positions:", "yes" if electrodes is not None else "none for this recording")
print("resection known:", "yes" if resection is not None else "no")
print("steps done to the signal:")
for step in steps:
    print("  -", step)
