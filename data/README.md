# Data Directory

CSV files here are gitignored. Download them from the UK Department for Transport:
https://www.data.gov.uk/dataset/road-accidents-safety-data

## File names

The loaders (`src/data_loader.py`) accept **any year** and either naming scheme.
If several years are present, they use the most recent one:

- short: `collision_2023.csv`, `vehicle_2023.csv`, `casualty_2023.csv`
- official: `dft-road-casualty-statistics-collision-2023.csv` (and `-vehicle-`, `-casualty-`)

## What needs what

| Command | Files needed |
|---------|--------------|
| `scripts/train_severity_leakfree.py`, `scripts/naive_baselines.py`, `scripts/data_insights.py`, `scripts/train_models.py --task timeseries` | collision only |
| `scripts/train_severity.py`, `scripts/train_models.py --task severity` (retrospective model) | collision + vehicle + casualty, same year |

## Files behind the README numbers

- **Full-year 2023 collision file**: 104,258 collisions (~19.8 MB). All re-run numbers in the
  main README (leakage-free severity, LSTM and naive baselines, data insights) come from it.
- **2025 provisional extract**: used for the SQL schema and the original write-up
  (collision 48,472 rows, vehicle 87,805, casualty 60,991). The retrospective severity
  score (macro-F1 ~0.76) came from this extract and has not been re-run.

## Key fields

- collision: date, time, location, weather, road conditions, `collision_severity`
- vehicle: vehicle type, driver age, manoeuvre
- casualty: `casualty_severity`, age, casualty type (post-crash; leaky for ahead-of-time prediction)
