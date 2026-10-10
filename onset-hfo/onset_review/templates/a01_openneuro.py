# Template: Fetch any recording from OpenNeuro
# Group: Start here
# Mirrors: File → Open from OpenNeuro…
# About: List a public dataset by its id, fetch a window of one recording with the dataset's own channel types and mains frequency, and run the detectors on it — the way scikit-learn fetches a dataset.

# %% What the dataset is
from onset_hfo import openneuro
from onset_hfo.datasets import OfflineError

DATASET = "ds003029"            # any OpenNeuro id; this one records at 1000 Hz
try:
    about = openneuro.describe_openneuro(DATASET)
    print(about.DESCR[:600])
    recordings = openneuro.list_openneuro(DATASET)
    print(recordings.drop(columns="path").head(10).to_string(index=False))
except (OfflineError, OSError, RuntimeError) as problem:
    about = recordings = None
    print(f"OpenNeuro could not be reached from here ({type(problem).__name__}), so this "
          "template stops. Downloads are kept under", openneuro.get_data_home())

# %% A window of one recording
bunch = None
if recordings is not None and len(recordings):
    first = recordings.iloc[0]
    bunch = openneuro.fetch_openneuro(DATASET, first["subject"], task=first["task"] or None,
                                      run=first["run"] or None, t_start=0, t_stop=60)
    print(bunch.data.shape, "channels x samples, volts;", f"{bunch.sfreq:g} Hz;",
          f"{bunch.times[0]:g}–{bunch.times[-1]:g} s of the original recording")
    print("mains", bunch.line_freq, "Hz; bad channels:", bunch.recording.bads[:10])
    print("cite:", bunch.citation)

# %% Run the detectors on it
if bunch is None:
    print("nothing fetched, so nothing to analyse")
elif bunch.sfreq <= 2 * 250:
    # Ripples reach 250 Hz, so the recording has to be sampled faster than 500 Hz.
    print(f"{bunch.sfreq:g} Hz is too slow for ripples (80–250 Hz need more than 500 Hz); "
          "pick a faster recording")
else:
    from onset_hfo.pipeline import run_pipeline

    result = run_pipeline(bunch.recording, detectors=("rms",), verbose=False)
    print(result.rates["rms"].head(10).to_string(index=False))
