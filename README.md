# AsymGAD

Repository for code related to "AsymGAD: Label-Free Lateral-Movement Pivot
Ranking via Asymmetric Graph Anomaly Detection".

AsymGAD is a label-free framework that turns a raw authentication stream into
a ranked list of lateral-movement pivot candidates.

## Installation

```
pip install -r requirements.txt
```

## Data

Download the LANL CMCSE data set (https://csr.lanl.gov/data/cyber1/) and
preprocess the raw files:

```
usage: python scripts/run_preprocess.py --raw-dir <lanl_raw_dir>
                                        [--config configs/lanl.json]

options:
  --raw-dir   Directory containing LANL CMCSE auth.txt / flows.txt / redteam.txt
```

## Window construction

```
usage: python scripts/run_window_construction.py [--config configs/lanl.json]
                                                 [--fixed-event]

usage: python scripts/run_window_stats.py [--config configs/lanl.json]
```

## Benchmark

```
usage: python scripts/run_benchmark.py [--config configs/lanl.json]
                                       [--limit N] [--epochs N] [--seeds 42,123,456]

options:
  --limit    Process at most this many windows (0 = all)
  --epochs   Override the number of training epochs
  --seeds    Comma-separated seed list; overrides the config
```

## Stability analysis

```
usage: python scripts/run_nonattack.py [--config configs/lanl.json]
                                       [--limit N] [--epochs N]
```

## Ablation studies

```
usage: python scripts/run_ablation.py [--config configs/lanl.json]
                                      [--limit N] [--epochs N]
                                      [--fixed-windows output/fixed_event_2M/windows.json]
```

## Tests

```
python tests/test_smoke.py
python tests/test_cleaning.py
```

## License

MIT License (LICENSE).
