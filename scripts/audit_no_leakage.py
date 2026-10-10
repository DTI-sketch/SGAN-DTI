#!/usr/bin/env python3
"""Construct selected folds and run the built-in DTI edge assertions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.load_graph import raw_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("data_luo", "data", "data_zeng"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fold-indices", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    folds = [int(value) for value in args.fold_indices.split(",") if value.strip()]
    for fold in folds:
        raw_graph(args.dataset, fold, dataset_dir=args.data_dir)
    print(f"PASS dataset={args.dataset} folds={folds}: no held-out DTI edges found")


if __name__ == "__main__":
    main()
