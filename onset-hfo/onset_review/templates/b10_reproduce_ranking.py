# Template: Reproduce the ranking, step by step
# Group: Detection and ranking
# Mirrors: the Recording page
# About: The window's whole analysis in plain calls -- preprocess, check quality, detect, validate, measure rates with intervals, rank, find the leader and its ties -- checked against the numbers on screen. Change any setting and run it again.

from onset_hfo.detectors import DETECTORS
from onset_hfo.metrics import channel_rates, leader_separation, rank_channels
from onset_hfo.outcome import candidate_channels
from onset_hfo.preprocess import prepare
from onset_hfo.quality import analysable_seconds, channel_quality, reject_unusable_events, segment_quality
from onset_hfo.validate import validate_events
from onset_review.session import laplacian_positions

# %% 1. The settings this window used (change them here to try others)
cfg = request.pipeline_config()
print("detectors:", request.detectors, "· band:", request.band_hz, "Hz")
print("ranking detector:", request.primary)

# %% 2. Preprocess the recording as the window did
positions, positions_from = laplacian_positions(request)
prep = prepare(recording, cfg.preprocess, verbose=False, positions=positions,
               positions_from=positions_from)
print(f"{len(prep.ch_names)} channels at {prep.sfreq:g} Hz")

# %% 3. Which contacts and which seconds are fit to analyse
segs = qual = None
if cfg.check_quality:
    segs = segment_quality(prep, cfg.quality)
    qual = channel_quality(prep, cfg.quality, segments=segs)
    print(f"{int((~qual['good']).sum())} channel(s) set aside")

# %% 4. Detect with each detector, and validate every event
found = []
for name in request.detectors:
    these = DETECTORS[name](prep, getattr(cfg, name))
    validate_events(these, prep, cfg.validation)
    found += list(these)
if request.with_spikes:
    found += list(DETECTORS["spike"](prep, cfg.spikes))
if cfg.check_quality:
    reject_unusable_events(found, segs, qual, keep=request.keep_channels)
print(f"{len(found)} events, {sum(e.accepted for e in found)} accepted")

# %% 5. Rates per channel over the seconds actually analysed, ranked
clean = analysable_seconds(segs, qual, list(prep.ch_names), prep.duration,
                           keep=request.keep_channels)
mine = channel_rates([e for e in found if e.detector == request.primary],
                     clean or prep.duration, channels=list(prep.ch_names))
mine = rank_channels(mine).reset_index(drop=True)
print(mine[["channel", "n_events", "rate_per_min", "rate_ci_low", "rate_ci_high"]]
      .head(10).to_string(index=False))

# %% 6. Does any channel stand out, and which are tied with it
my_leader = leader_separation(mine)
counts = mine.set_index("channel")["n_events"]
my_tied = candidate_channels(counts, max(prep.duration / 60.0, 1e-9))
print(my_leader["statement"])
print("tied with the busiest:", my_tied)

# %% 7. The same numbers as the window?
same_rates = np.allclose(mine["rate_per_min"].to_numpy(), findings["rate_per_min"].to_numpy())
same_order = list(mine["channel"]) == list(findings["channel"])
print("rates match the window:", same_rates, "· order matches:", same_order)
print("leader matches:", my_leader.get("leader") == leader.get("leader"),
      "· tied set matches:", list(my_tied) == list(candidates))
