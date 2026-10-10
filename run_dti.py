"""Train and evaluate the leakage-free SGAN-DTI implementation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Must be defined before the first CUDA/cuBLAS operation when deterministic
# algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef

from model.MAGNN_dti import MAGNN_lp
from utils.data_loader import DataLoader
from utils.load_graph import raw_graph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_threshold(confidence, labels, objective: str) -> float:
    """Select one classification threshold using validation data only."""
    scores = np.asarray(confidence)[:, 1]
    labels = np.asarray(labels, dtype=np.int64)
    best_value = -np.inf
    best_threshold = 0.5
    for threshold in np.linspace(0.0, 1.0, 101):
        prediction = (scores >= threshold).astype(np.int64)
        if np.unique(prediction).size < 2:
            continue
        f1 = f1_score(labels, prediction, zero_division=0)
        mcc = matthews_corrcoef(labels, prediction)
        value = mcc if objective == "mcc" else (f1 + mcc) / 2.0
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def _identity_features(graph, device: torch.device):
    features = []
    dimensions = []
    for node_type in graph.ntypes:
        size = graph.num_nodes(node_type)
        indices = torch.arange(size, dtype=torch.long).repeat(2, 1)
        values = torch.ones(size, dtype=torch.float32)
        features.append(
            torch.sparse_coo_tensor(indices, values, (size, size)).coalesce().to(device)
        )
        dimensions.append(size)
    return features, dimensions


def _homogeneous_graph(graph, device: torch.device):
    homogeneous = dgl.to_homogeneous(graph)
    homogeneous = dgl.remove_self_loop(homogeneous)
    original_edges = homogeneous.num_edges()
    homogeneous = dgl.add_self_loop(homogeneous)
    homogeneous.edata["_ID"][original_edges:] = torch.arange(
        homogeneous.num_edges() - original_edges
    )
    homogeneous.edata["_TYPE"][original_edges:] = len(graph.etypes)
    return homogeneous.to(device), len(graph.etypes) + 1


def prepare_inputs(dataset: str, dataset_dir: Path, fold: int, device: torch.device):
    drug_channel, target_channel = raw_graph(
        dataset, fold, task="train", dataset_dir=dataset_dir
    )
    drug_features, drug_dimensions = _identity_features(drug_channel, device)
    target_features, target_dimensions = _identity_features(target_channel, device)
    drug_graph, drug_edge_types = _homogeneous_graph(drug_channel, device)
    target_graph, target_edge_types = _homogeneous_graph(target_channel, device)
    return {
        "drug_graph": drug_graph,
        "target_graph": target_graph,
        "drug_features": drug_features,
        "target_features": target_features,
        "drug_dimensions": drug_dimensions,
        "target_dimensions": target_dimensions,
        "drug_edge_features": drug_graph.edata["_TYPE"],
        "target_edge_features": target_graph.edata["_TYPE"],
        "drug_edge_types": drug_edge_types,
        "target_edge_types": target_edge_types,
    }


def model_inputs(prepared, pairs: np.ndarray, device: torch.device):
    left = pairs[0]
    right = pairs[1]
    relation = np.zeros(pairs.shape[1], dtype=np.int32)
    dense_pair_graph = torch.ones((pairs.shape[1], pairs.shape[1]), device=device)
    return (
        prepared["drug_graph"],
        prepared["target_graph"],
        prepared["drug_features"],
        prepared["target_features"],
        prepared["drug_edge_features"],
        prepared["target_edge_features"],
        left,
        right,
        relation,
        dense_pair_graph,
    )


def labelled_pairs(positive: np.ndarray, negative: np.ndarray):
    pairs = np.concatenate((positive, negative), axis=1)
    labels = np.concatenate(
        (np.ones(positive.shape[1]), np.zeros(negative.shape[1]))
    ).astype(np.int64)
    return pairs, labels


@torch.no_grad()
def predict(model, prepared, pairs, device):
    model.eval()
    return model(model_inputs(prepared, pairs, device)).cpu().numpy()


def metric_summary(rows):
    metrics = ("auc", "aupr", "precision", "recall", "F1", "MCC")
    summary = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def run_fold(args, fold: int, device: torch.device, output_dir: Path):
    set_seed(args.seed)
    loader = DataLoader(args.data_dir, fold, args.fold_num)
    prepared = prepare_inputs(args.dataset, args.data_dir, fold, device)

    heads = [args.num_heads] * (args.num_layers + 1)
    model = MAGNN_lp(
        args.edge_feats,
        prepared["drug_edge_types"],
        prepared["target_edge_types"],
        prepared["drug_dimensions"],
        prepared["target_dimensions"],
        args.hidden_dim,
        args.hidden_dim,
        args.num_layers,
        heads,
        F.elu,
        args.dropout,
        args.dropout,
        args.slope,
        args.residual,
        args.residual_att,
        args.decoder,
        args,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    checkpoint = output_dir / "checkpoints" / f"fold_{fold}_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    train_positive = loader.train_pos
    train_negative = loader.train_neg
    validation_pairs, validation_labels = labelled_pairs(loader.valid_pos, loader.valid_neg)
    best_validation_aupr = -np.inf
    stale_epochs = 0
    epochs_ran = 0

    for epoch in range(args.epochs):
        model.train()
        positive_order = np.random.permutation(train_positive.shape[1])
        negative_order = np.random.permutation(train_negative.shape[1])
        epoch_losses = []
        epoch_labels = []
        epoch_predictions = []
        for start in range(0, train_positive.shape[1], args.batch_size):
            stop = min(start + args.batch_size, train_positive.shape[1])
            positive = train_positive[:, positive_order[start:stop]]
            negative = train_negative[:, negative_order[start:stop]]
            pairs, labels = labelled_pairs(positive, negative)
            label_tensor = torch.from_numpy(labels).long().to(device)

            probability = model(model_inputs(prepared, pairs, device))
            loss = F.nll_loss(torch.log(probability.clamp_min(1e-12)), label_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            epoch_labels.append(labels)
            epoch_predictions.append(probability.detach().cpu().numpy())

        validation_scores = predict(model, prepared, validation_pairs, device)
        validation_threshold = select_threshold(
            validation_scores, validation_labels, args.threshold_objective
        )
        validation_metrics = loader.evaluate(
            validation_scores, validation_labels, validation_threshold
        )
        epochs_ran = epoch + 1
        train_f1 = f1_score(
            np.concatenate(epoch_labels),
            np.concatenate(epoch_predictions).argmax(axis=1),
            zero_division=0,
        )
        print(
            f"fold={fold} epoch={epoch:03d} loss={np.mean(epoch_losses):.6f} "
            f"train_f1={train_f1:.6f} val_aupr={validation_metrics['aupr']:.6f}",
            flush=True,
        )
        if validation_metrics["aupr"] > best_validation_aupr + args.early_stop_delta:
            best_validation_aupr = validation_metrics["aupr"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    validation_scores = predict(model, prepared, validation_pairs, device)
    threshold = select_threshold(
        validation_scores, validation_labels, args.threshold_objective
    )
    validation_metrics = loader.evaluate(validation_scores, validation_labels, threshold)
    validation_metrics["threshold"] = threshold

    test_pairs, test_labels = labelled_pairs(loader.test_pos, loader.test_neg)
    test_scores = predict(model, prepared, test_pairs, device)
    test_metrics = loader.evaluate(test_scores, test_labels, threshold)
    test_metrics["threshold"] = threshold
    loader.write_predictions(
        test_pairs,
        test_scores[:, 1],
        test_labels,
        output_dir / f"fold_{fold}_test_predictions.tsv",
    )
    result = {
        "fold": fold,
        "seed": args.seed,
        "epochs_ran": epochs_ran,
        "graph_policy": "training-positive DTI edges only",
        "validation_positive_edges_in_graph": 0,
        "test_positive_edges_in_graph": 0,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    with (output_dir / f"fold_{fold}.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("data_luo", "data", "data_zeng"), required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset directory containing 10_folds/ and matrix files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold-num", type=int, default=10)
    parser.add_argument("--fold-indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--edge-feats", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--slope", type=float, default=0.01)
    parser.add_argument("--residual-att", type=float, default=0.5)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder", default="dot")
    parser.add_argument("--predictor", choices=("gcn", "linear"), default="gcn")
    parser.add_argument("--semantic-fusion", choices=("concatenation",), default="concatenation")
    parser.add_argument(
        "--threshold-objective", choices=("mcc", "f1_mcc"), default="f1_mcc"
    )
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir / args.dataset).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu for checks")
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    folds = [int(value) for value in args.fold_indices.split(",") if value.strip()]
    if not folds or any(fold < 0 or fold >= args.fold_num for fold in folds):
        raise ValueError("fold indices must lie in [0, fold_num)")

    started = time.perf_counter()
    results = [run_fold(args, fold, device, output_dir) for fold in folds]
    summary = {
        "status": "complete",
        "dataset": args.dataset,
        "folds": folds,
        "seed_reset_each_fold": args.seed,
        "protocol": "fixed 80/10/10; validation-only selection",
        "graph_policy": "training-positive DTI edges only",
        "validation_positive_edges_in_graph": 0,
        "test_positive_edges_in_graph": 0,
        "validation": metric_summary([row["validation"] for row in results]),
        "test": metric_summary([row["test"] for row in results]),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
