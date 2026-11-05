# UK Road Casualty Severity Analysis

Analysis of UK Department for Transport road-safety data: casualty severity, pre-incident collision severity and accident timing. The retrospective casualty model uses information known after a crash; only the separate collision-level model uses pre-incident features. Neither is a real-time safety system.

## Quick demo

```bash
pip install -r requirements.txt
python scripts/quick_demo.py
```

## Real data

Download the DfT files listed in [`data/README.md`](data/README.md). Train the pre-incident model with:

```bash
python scripts/train_severity_leakfree.py --model lr
```

The previously published full-data scores predate a missing-value preprocessing correction and need to be rerun before use. The leakage-free model's earlier macro-F1 was about 0.35. See the training scripts for current metrics and `sql/` for analytics queries.

## Tests

```bash
python -m pytest -q
```

SQL setup requires PostgreSQL; the synthetic demo and tests do not require downloaded DfT data.
