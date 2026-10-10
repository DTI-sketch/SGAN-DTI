# Bundled runtime data

Only files consumed by `utils/data_loader.py` and `utils/load_graph.py` are
included. Each dataset contains:

- `10_folds/pos_folds.json`;
- `10_folds/neg_folds_times_1.json`; and
- six relation matrices: drug--target, drug--drug, drug--disease,
  drug--side-effect, target--target, and target--disease.

These files reproduce the fixed 80%/10%/10% experiments. See the repository
root [DATASETS.md](../DATASETS.md) for provenance and redistribution cautions.

