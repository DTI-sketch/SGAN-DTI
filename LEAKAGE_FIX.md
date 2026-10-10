# DTI edge-leakage correction

The original heterogeneous matrices contain all known positive drug-target
interactions. If they are used directly for message passing after the pairs
have been divided into train, validation, and test subsets, validation/test
positive labels remain visible to the encoder.

The corrected pipeline applies the following order:

1. Read the fixed ten-fold positive and negative pair files.
2. Use eight folds for training, the next fold for validation, and the current
   fold for testing (80/10/10).
3. Construct the heterogeneous graph and preserve all permitted non-DTI
   auxiliary relations.
4. Remove validation and test positive DTI edges from both `d_p` and `p_d`.
5. Assert that the remaining forward and reverse DTI relations are exactly the
   training-positive pairs.
6. Use this same frozen training-only message-passing graph during training,
   validation, and testing.

The checks are executed every time `raw_graph` is called. A run stops before
training if any held-out positive edge remains.

Relevant implementation:

- `utils/load_graph.py::remove_heldout_dti_edges`
- `utils/load_graph.py::assert_training_only_dti`
- `tests/test_no_leakage.py`
- `scripts/audit_no_leakage.py`
