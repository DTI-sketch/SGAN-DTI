# SGAN-DTI

Leakage-free implementation of **SGAN-DTI: A Dual Channel Subgraph Attention
Framework for Drug--Target Interaction Prediction by Eliminating Meta-path
Dependency**.

SGAN-DTI contains two relation-aware heterogeneous subgraph encoders and a
batch-defined drug--target-pair GCN (DTP-GCN) predictor. This release preserves
that architecture and corrects the DTI edge-leakage problem in graph
construction. Chinese documentation is available in [README_zh.md](README_zh.md).

> **Scope note:** this is a cleaned, standalone reproduction implementation,
> not a line-by-line copy of the historical training script. It retains the
> SGAN-DTI model design and evaluation behavior needed for the reported run,
> while reorganising data loading, output handling, and reproducibility checks.
> The scientific correction is that held-out DTI labels are no longer present
> in the message-passing graph.

## What is included

- Training and evaluation code for the original SGAN-DTI architecture.
- The three fixed benchmark datasets used by the supplied scripts.
- Materialised 10-fold positive and 1:1 negative pair files.
- Runtime assertions and regression tests for DTI edge leakage.
- Reproduction scripts for all three datasets.
- Per-fold metrics and predictions from the verified 2026-10-09 reference run.

Checkpoints are not included because they can be regenerated and would add
about 239 MB. The bundled data are third-party research data and are **not**
covered by the code license; see [DATASETS.md](DATASETS.md).

## Leakage-free evaluation protocol

For test fold `k`, fold `(k + 1) mod 10` is validation data and the remaining
eight folds are training data, giving a fixed 80%/10%/10% split. Before any
message passing:

1. validation and test positive DTI edges are removed from both `d_p` and
   `p_d`;
2. the remaining forward and reverse DTI relations are asserted to equal the
   training-positive edge set exactly; and
3. the same frozen training-only graph is used in training, validation, and
   testing.

Non-DTI auxiliary relations are retained. Model selection, early stopping, and
classification-threshold selection use validation data only. The test set is
evaluated once using the selected model and fixed threshold. See
[LEAKAGE_FIX.md](LEAKAGE_FIX.md) for implementation details.

## Repository layout

```text
.
├── datasets/                    # required matrices and fixed fold files
├── model/                       # dual-channel encoder and DTP-GCN
├── reference_results/           # verified 10-fold metrics and predictions
├── scripts/
│   ├── audit_no_leakage.py
│   └── run_three_datasets.sh
├── tests/test_no_leakage.py
├── utils/
│   ├── data_loader.py
│   └── load_graph.py
├── run_dti.py
├── requirements.txt
└── requirements-cu118.txt
```

## Environment

The reference run used:

- Python 3.9.23
- PyTorch 2.3.1 + CUDA 11.8
- DGL 2.4.0 + CUDA 11.8
- NumPy 1.26.3
- SciPy 1.13.1
- scikit-learn 1.6.1

Create a clean environment and install the verified CUDA 11.8 stack:

```bash
conda create -n sgan-dti python=3.9 -y
conda activate sgan-dti
pip install -r requirements-cu118.txt
```

For a different CUDA or CPU environment, install mutually compatible PyTorch
and DGL builds first, then run:

```bash
pip install -r requirements.txt
```

The verified PyTorch command follows the official
[previous-version instructions](https://pytorch.org/get-started/previous-versions/).

## Verify before training

Run the small regression test:

```bash
python -m unittest tests.test_no_leakage
```

Audit all real folds:

```bash
python scripts/audit_no_leakage.py \
  --dataset data_luo \
  --data-dir datasets/data_luo \
  --fold-indices 0,1,2,3,4,5,6,7,8,9
```

A successful audit prints
`validation_in_graph=0 test_in_graph=0` for every fold. Any mismatch raises an
exception before training starts.

## Reproduce all experiments

Run all ten folds of all three datasets serially on one GPU:

```bash
bash scripts/run_three_datasets.sh datasets cuda:0
```

Run one dataset manually, for example Hetero-A:

```bash
python -u run_dti.py \
  --dataset data_luo \
  --data-dir datasets/data_luo \
  --output-dir outputs \
  --device cuda:0 \
  --seed 3407 \
  --fold-indices 0,1,2,3,4,5,6,7,8,9 \
  --hidden-dim 32 \
  --num-heads 1 \
  --num-layers 3 \
  --lr 5e-4 \
  --dropout 0.5 \
  --weight-decay 2e-4 \
  --batch-size 128 \
  --predictor gcn \
  --semantic-fusion concatenation \
  --threshold-objective f1_mcc
```

Each fold writes metrics, test predictions, and its best validation checkpoint.
`summary.json` contains the mean and sample standard deviation across folds.


## Dataset identifiers

| Code identifier | Manuscript name |
|---|---|
| `data_luo` | Hetero-A |
| `data` | Hetero-B |
| `data_zeng` | Zeng's dataset |

## Citation

The manuscript is currently under review. Citation metadata in
[CITATION.cff](CITATION.cff) intentionally uses the group author name and can
be replaced with the final bibliographic record after publication.

## License and third-party material

The SGAN-DTI code in this repository is released under the MIT License; see
[LICENSE](LICENSE). Dataset files remain subject to their original database and
publication terms. `model/conv.py` is adapted from DGL's GAT convolution code;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the bundled Apache 2.0
license.
