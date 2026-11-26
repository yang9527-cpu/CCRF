# CCRF

PyTorch implementation of CCRF for aligned class probabilities from multiple sources. Training uses source probabilities without reference labels.

## Requirements

- Python 3.11 or 3.12
- NumPy 1.26.4
- PyTorch 2.9.1

```bash
pip install -r requirements.txt
```

## Files

```text
CCRF/
├── run_CCRF.py
├── CCRF_Solver.py
├── CCRF_update.py
├── measures/
│   └── evaluate.py
├── utils/
│   └── data.py
├── requirements.txt
└── README.md
```

`run_CCRF.py` provides training, prediction and evaluation. `CCRF_Solver.py` contains the model and optimizer. `CCRF_update.py` implements group association and source mutual information.

## Example

```bash
python run_CCRF.py demo --device cpu
```

This creates synthetic inputs and runs training, prediction and evaluation in `demo_run/`. It checks the installation and does not reproduce benchmark results.

## Data

Store each fitting or evaluation input in an NPZ file with the following arrays:

| Array | Shape | Description |
|---|---|---|
| `P` | `(N, M, K)` | Floating-point probabilities; each source vector sums to one |
| `sample_id` | `(N,)` | Unique image identifiers |
| `source_names` | `(M,)` | Unique source names in the source-axis order |
| `class_names` | `(K,)` | Unique class names in the class-axis order |

At least three sources and two classes are required. The class and source orders must match between fitting and prediction. Source arrays named `model_ids` or `models` are also accepted. Reference labels are stored separately as integer `Y` with shape `(N,)` and the same ordered `sample_id` array. Optional `class_names` in the label file must match the prediction vocabulary.

## Training and prediction

```bash
python run_CCRF.py fit --input train.npz --checkpoint runs/CCRF.pt --device cuda
python run_CCRF.py predict --input test.npz --checkpoint runs/CCRF.pt --output runs/prediction.npz --device cuda
python run_CCRF.py evaluate --prediction runs/prediction.npz --labels labels.npz --sources test.npz --output runs/metrics.json
```

Use `--device cpu` for CPU execution or `--device auto` to select an available device. Defaults are 80 epochs, hidden width 64, batch limit 1024 and seed 2026. Fitting and prediction apply the same input representation. Checkpoints retain the model configuration, fitting sample identifiers and ordered class/source names.

Inductive evaluation fits on the unlabeled training subset and predicts a separate held-out subset. Transductive evaluation fits and predicts on the same unlabeled pool. Prepare and reuse the same saved split for every method being compared.

## Variants

Use `--method` with one of the following names:

| Method | Configuration |
|---|---|
| `CCRF` | Complete source responses and both objectives |
| `Top1` | Winning class identities only |
| `Top1Conf` | Winning classes and confidences |
| `FullLinear` | Full linear residual instead of the nonlinear residual |
| `NoSourceMIHalf` | Group association only, retaining its coefficient of 0.5 |

For an objective-weight experiment, `--gamma` varies the source-MI coefficient while keeping the group-association coefficient at 0.5. The default is 1. This differs from a loss normalized by `1 + gamma` when gamma is not 1.

## Evaluation

Evaluation reports accuracy, macro-F1, Brier score multiplied by 100 and NLL. Supplying `--sources` also reports coverage, recovery and damage. Conditional recovery uses the images missed by every source; conditional damage uses those with at least one correct source. Reference labels are opened only by evaluation.

The repository contains the CCRF implementation and the listed controls. Empirical source probabilities, VLM weights and third-party baseline implementations are separate from this code. Exact benchmark reproduction requires the corresponding probability archives, class dictionaries, sample identifiers and splits. Preserve input precision when using the compressed input controls; casting to float32 before representation can change a training trajectory.
