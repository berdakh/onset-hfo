# Template: Try another preprocessing
# Group: Signal
# Mirrors: the Preprocessing panel
# About: The same recording with one preprocessing choice changed -- here the reference -- and what that does to the ranking. The window's own analysis is left untouched.

from dataclasses import replace

from onset_hfo.config import PreprocessConfig
from onset_hfo.detectors import DETECTORS
from onset_hfo.metrics import channel_rates, rank_channels
from onset_hfo.preprocess import effective_reference, prepare
from onset_hfo.validate import validate_events

# %% What the window used
cfg = request.pipeline_config()
base = cfg.preprocess or PreprocessConfig()
print("this window:", request.preprocess_label())


def ranking(preprocess):
    prep = prepare(recording, preprocess, verbose=False)
    found = DETECTORS[request.primary](prep, getattr(cfg, request.primary))
    validate_events(found, prep, cfg.validation)
    table = rank_channels(channel_rates(found, prep.duration, channels=list(prep.ch_names)))
    return prep, table


# %% Change one thing: the reference ("bipolar", "average", "median", "shaft", "laplacian", "none")
THIS_REFERENCE = effective_reference(base)
OTHER_REFERENCE = "shaft" if THIS_REFERENCE != "shaft" else "bipolar"
prep_a, ranked_a = ranking(base)
prep_b, ranked_b = ranking(replace(base, reference=OTHER_REFERENCE))
print(f"{THIS_REFERENCE}: top 5", list(ranked_a["channel"].head(5)))
print(f"{OTHER_REFERENCE}: top 5", list(ranked_b["channel"].head(5)))

# %% What was done, step by step
for step in prep_b.steps:
    print("  -", step)
