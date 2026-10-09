# Template: Which contacts and seconds were fit to analyse
# Group: Signal
# Mirrors: the Signal page
# About: The quality checks the window ran before detecting anything -- amplitude, clipping, mains, high-frequency share, bursts -- per channel and per segment, recomputed and compared with the window's tables.

from onset_hfo.preprocess import prepare
from onset_hfo.quality import channel_quality, clean_seconds, segment_quality

# %% Recompute both tables from the recording
cfg = request.pipeline_config()
prep = prepare(recording, cfg.preprocess, verbose=False)
segs = segment_quality(prep, cfg.quality)
qual = channel_quality(prep, cfg.quality, segments=segs)
print(qual[["channel", "amplitude_uv", "line_fraction", "hf_ratio", "good", "reason"]]
      .to_string(index=False))

# %% The same as the window's?
if quality is not None:
    print("channel verdicts match:", list(qual["good"]) == list(quality["good"]))

# %% Seconds kept per channel: what every rate is divided by
kept = clean_seconds(segs, list(prep.ch_names))
print(pd.Series(kept, name="seconds analysed").round(1).to_string())

# %% Which segments were rejected, and why
bad = segs[~segs["good"]]
print(f"{len(bad)} of {len(segs)} channel-segments rejected")
print(bad["reason"].value_counts().to_string() if len(bad) else "none")
