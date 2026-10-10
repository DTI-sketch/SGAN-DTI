# Reference results

`2026-10-09/` contains the complete fold-level JSON metrics and test prediction
files from the verified leakage-free run. Checkpoints are excluded.

The run used fixed folds 0--9, seed 3407 reset for each fold, validation-only
early stopping and threshold selection, and training-positive DTI edges only in
both message-passing channels. Each dataset's `summary.json` records the exact
configuration and aggregate metrics using repository-relative paths.

