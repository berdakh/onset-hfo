# Template: Save the results as files
# Group: Results
# Mirrors: the Report's tables
# About: The ranking, every event, the quality tables and what was done to the signal, written as CSV and text into the Files pane's folder (the console's working directory) for analysis elsewhere.

from onset_hfo.detectors.base import events_to_frame

# %% Where to write (the Files pane's folder, unless you change it)
OUT = Path.cwd() / f"{request.subject}_{request.t_start:g}-{request.t_stop:g}s"
OUT.mkdir(parents=True, exist_ok=True)

# %% Write
findings.to_csv(OUT / "ranking.csv", index=False)
events_to_frame(list(events)).to_csv(OUT / "events.csv", index=False)
if quality is not None:
    quality.to_csv(OUT / "channel_quality.csv", index=False)
if segments is not None:
    segments.to_csv(OUT / "segment_quality.csv", index=False)
(OUT / "steps.txt").write_text(request.label() + "\n" + "\n".join(steps) + "\n")
for path in sorted(OUT.iterdir()):
    print(f"{path.name:>22}  {path.stat().st_size:>9,} bytes")
