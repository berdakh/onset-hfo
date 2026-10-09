# Template: Epileptogenicity Index around a seizure
# Group: Seizure onset
# Mirrors: a case's Ictal onset step
# About: The Epileptogenicity Index (Bartolomei 2008) of every channel for one seizure: the fast-over-slow energy ratio, a Page-Hinkley change test, and how early each channel changed. Set SEIZURE_ONSET_S to the electrographic onset you marked.

from onset_hfo.ictal import EI_CUTOFF, IctalSettings, epileptogenicity

# %% Where the seizure starts, in recording seconds
SEIZURE_ONSET_S = None          # e.g. 1234.5; None looks for a marked seizure in this window
settings = IctalSettings()      # the method's own settings, untuned
if SEIZURE_ONSET_S is None:
    marked = [a["onset"] for a in raw.annotations
              if "seizure" in str(a["description"]).lower()]
    if marked:
        SEIZURE_ONSET_S = float(marked[0]) + t_offset
        print(f"using the seizure marked at {SEIZURE_ONSET_S:.1f} s")
    else:
        # No seizure here: the middle of the window, which is what the
        # validation's control does -- expect no real onset to be found.
        SEIZURE_ONSET_S = t_offset + raw.times[-1] / 2
        print(f"no seizure is marked in this window; using its middle "
              f"({SEIZURE_ONSET_S:.1f} s) as a control, not an onset")
print(settings.describe())

# %% The index of every channel
before = SEIZURE_ONSET_S - t_offset
after = t_offset + raw.times[-1] - SEIZURE_ONSET_S
table = None
if before < settings.pre_s or after < settings.post_s:
    print(f"This window has {before:.0f} s before that time and {after:.0f} s after it; the "
          f"index needs {settings.pre_s:g} s of baseline before the onset and "
          f"{settings.post_s:g} s after it. Open a window that starts at least "
          f"{settings.pre_s:g} s before the seizure (Recording → Next window, or File → "
          "Open a recording…), or a case's Ictal onset step, which reads around each "
          "marked seizure for you.")
else:
    table = epileptogenicity(np.asarray(signal), list(channels), sfreq,
                             onset_s=SEIZURE_ONSET_S, settings=settings, start_s=t_offset)
    print(table.head(10).to_string(index=False))
    print(f"{int((table['ei'] >= EI_CUTOFF).sum())} channel(s) at or above {EI_CUTOFF}")

# %% Plot the index per channel
if table is not None:
    shown = table.sort_values("ei", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(shown) + 1))
    ax.barh(shown["channel"], shown["ei"])
    ax.axvline(EI_CUTOFF, linestyle="--", color="0.4")
    ax.set_xlabel("Epileptogenicity Index")
    fig.tight_layout()
    plt.show()
