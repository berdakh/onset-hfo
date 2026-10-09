# Template: Compare the detectors
# Group: Detection and ranking
# Mirrors: the Recording page's detector columns
# About: Every HFO detector on the same preprocessed signal, how many events each finds, how often they agree event by event (Jaccard), and whether they rank the same channels first.

from onset_hfo.detectors import HFO_DETECTORS
from onset_hfo.metrics import agreement_matrix, channel_rates, rank_channels
from onset_hfo.preprocess import prepare
from onset_hfo.validate import validate_events

# %% Preprocess once, detect with every HFO detector
cfg = request.pipeline_config()
prep = prepare(recording, cfg.preprocess, verbose=False)
by_detector = {}
for name, detect in HFO_DETECTORS.items():
    these = detect(prep, getattr(cfg, name))
    validate_events(these, prep, cfg.validation)
    by_detector[name] = [e for e in these if e.accepted]
    print(f"{name:>18}: {len(by_detector[name])} accepted events")

# %% Event-by-event agreement (1.0 on the diagonal by construction)
print(agreement_matrix(by_detector).round(2).to_string())

# %% Do they put the same channels first?
tops = {}
for name, found in by_detector.items():
    ranked = rank_channels(channel_rates(found, prep.duration, channels=list(prep.ch_names)))
    tops[name] = list(ranked["channel"].head(5))
print(pd.DataFrame(tops).to_string())
