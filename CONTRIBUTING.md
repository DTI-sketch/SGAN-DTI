# Contributing

Contributions are welcome through GitHub issues and pull requests.

Before submitting a change:

1. run `python -m unittest tests.test_no_leakage`;
2. run `scripts/audit_no_leakage.py` on every affected dataset and fold;
3. do not introduce validation or test DTI edges into a message-passing graph;
4. do not use test metrics for model, parameter, checkpoint, or threshold
   selection; and
5. document changes that alter splits, negative sampling, features, or metrics.

Please keep generated checkpoints, logs, and local absolute paths out of pull
requests.

