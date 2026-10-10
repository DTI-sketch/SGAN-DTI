"""Dataset and fixed-fold loading for SGAN-DTI.

The loader reads the public matrix files and the materialised ten-fold pair
files only.  A fold is used as test data, the following fold as validation
data, and the remaining eight folds as training data (80/10/10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


class DataLoader:
    """Load one fixed 80/10/10 fold without constructing a label graph."""

    def __init__(self, path: Union[str, Path], fold_index: int, fold_count: int = 10):
        self.path = Path(path).expanduser().resolve()
        self.data = self.path.name
        self.fold_index = int(fold_index)
        self.fold_count = int(fold_count)
        if not 0 <= self.fold_index < self.fold_count:
            raise ValueError("fold_index must be in [0, fold_count)")

        matrix_dir = self.path if (self.path / "mat_drug_protein.txt").exists() else self.path / "mat_data"
        drug_target = np.loadtxt(matrix_dir / "mat_drug_protein.txt", dtype=np.int8)
        drug_disease = np.loadtxt(matrix_dir / "mat_drug_disease.txt", dtype=np.int8)
        drug_side_effect = np.loadtxt(matrix_dir / "mat_drug_se.txt", dtype=np.int8)
        self.drug_num = int(drug_target.shape[0])
        self.target_num = int(drug_target.shape[1])
        self.node_counts = {
            "drug": self.drug_num,
            "target": self.target_num,
            "disease": int(drug_disease.shape[1]),
            "side_effect": int(drug_side_effect.shape[1]),
        }

        fold_dir = self.path / "10_folds"
        with (fold_dir / "pos_folds.json").open("r", encoding="utf-8") as stream:
            self._positive_folds = json.load(stream)
        with (fold_dir / "neg_folds_times_1.json").open("r", encoding="utf-8") as stream:
            self._negative_folds = json.load(stream)

        self.train_pos, self.valid_pos, self.test_pos = self._split(self._positive_folds)
        self.train_neg, self.valid_neg, self.test_neg = self._split(self._negative_folds)
        self._audit_splits()

    def _as_global_pairs(self, values) -> np.ndarray:
        pairs = np.asarray(values, dtype=np.int64).copy()
        if pairs.ndim != 2 or pairs.shape[0] != 2:
            raise ValueError(f"expected pair array with shape (2, N), got {pairs.shape}")
        pairs[1] += self.drug_num
        return pairs

    def _split(self, folds: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        test_fold = self.fold_index
        validation_fold = (self.fold_index + 1) % self.fold_count
        train = [
            np.asarray(folds[f"fold_{index}"], dtype=np.int64)
            for index in range(self.fold_count)
            if index not in (test_fold, validation_fold)
        ]
        train_pairs = self._as_global_pairs(np.concatenate(train, axis=1))
        validation_pairs = self._as_global_pairs(folds[f"fold_{validation_fold}"])
        test_pairs = self._as_global_pairs(folds[f"fold_{test_fold}"])
        return train_pairs, validation_pairs, test_pairs

    @staticmethod
    def _pair_set(pairs: np.ndarray) -> set[tuple[int, int]]:
        return set(map(tuple, pairs.T.tolist()))

    def _audit_splits(self) -> None:
        for name, (train, validation, test) in {
            "positive": (self.train_pos, self.valid_pos, self.test_pos),
            "negative": (self.train_neg, self.valid_neg, self.test_neg),
        }.items():
            train_set = self._pair_set(train)
            validation_set = self._pair_set(validation)
            test_set = self._pair_set(test)
            if train_set & validation_set or train_set & test_set or validation_set & test_set:
                raise AssertionError(f"{name} train/validation/test pairs are not disjoint")

    # Compatibility methods retained for the original training interface.
    def get_train_valid_test_pos(self, _fold_count: int, _fold_index: int):
        return self.train_pos.copy(), self.valid_pos.copy(), self.test_pos.copy()

    def get_train_neg(self, _fold_count: int, _fold_index: int):
        return self.train_neg.copy()

    def get_val_neg(self, _fold_count: int, _fold_index: int):
        return self.valid_neg.copy()

    def get_test_neg(self, _fold_count: int, _fold_index: int):
        return self.test_neg.copy()

    @staticmethod
    def evaluate(confidence, labels, threshold: float) -> dict[str, float]:
        scores = np.asarray(confidence)[:, 1]
        labels = np.asarray(labels, dtype=np.int64)
        prediction = (scores >= threshold).astype(np.int64)
        return {
            "auc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores)),
            "precision": float(precision_score(labels, prediction, zero_division=0)),
            "recall": float(recall_score(labels, prediction, zero_division=0)),
            "F1": float(f1_score(labels, prediction, zero_division=0)),
            "MCC": float(matthews_corrcoef(labels, prediction)),
        }

    @staticmethod
    def write_predictions(
        edge_list, confidence, labels, file_path: Union[str, Path]
    ) -> None:
        destination = Path(file_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for drug, target, label, score in zip(
                edge_list[0], edge_list[1], labels, confidence
            ):
                stream.write(f"{drug}\t{target}\t{int(label)}\t{float(score):.10f}\n")


# Preserve the import name used by the original project.
data_loader = DataLoader
