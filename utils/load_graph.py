"""Leakage-free heterogeneous graph construction.

The full biomedical graph is loaded for its non-DTI auxiliary relations.  DTI
edges belonging to validation or test pairs are then removed in both
directions before either encoder is called.  Runtime assertions require the
remaining DTI relation to equal the training-positive relation exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import dgl
import numpy as np
import scipy.sparse as sp
import torch

from utils.data_loader import DataLoader


DRUG = "0-drug"
TARGET = "1-protein"
DISEASE = "2-disease"
SIDE_EFFECT = "3-side-effect"


def _edge_set(graph: dgl.DGLHeteroGraph, edge_type: str) -> set:
    source, target = graph.edges(etype=edge_type)
    return set(zip(source.cpu().tolist(), target.cpu().tolist()))


def remove_heldout_dti_edges(
    graph: dgl.DGLHeteroGraph,
    validation_pairs: np.ndarray,
    test_pairs: np.ndarray,
) -> None:
    """Remove validation/test DTI edges from both directed relations."""
    for heldout in (validation_pairs, test_pairs):
        if heldout.shape[1] == 0:
            continue
        forward_ids = graph.edge_ids(heldout[0], heldout[1], etype="d_p")
        reverse_ids = graph.edge_ids(heldout[1], heldout[0], etype="p_d")
        graph.remove_edges(forward_ids, etype="d_p")
        graph.remove_edges(reverse_ids, etype="p_d")


def assert_training_only_dti(
    graph: dgl.DGLHeteroGraph,
    train_pairs: np.ndarray,
    validation_pairs: np.ndarray,
    test_pairs: np.ndarray,
    graph_name: str,
) -> None:
    """Fail immediately if a held-out positive edge remains in the graph."""
    expected_train = set(map(tuple, train_pairs.T.tolist()))
    validation = set(map(tuple, validation_pairs.T.tolist()))
    test = set(map(tuple, test_pairs.T.tolist()))
    forward = _edge_set(graph, "d_p")
    reverse = _edge_set(graph, "p_d")

    if forward != expected_train:
        missing = len(expected_train - forward)
        unexpected = len(forward - expected_train)
        raise AssertionError(
            f"{graph_name}: DTI graph differs from training positives "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if reverse != {(target, drug) for drug, target in expected_train}:
        raise AssertionError(f"{graph_name}: reverse DTI relation is inconsistent")
    if forward & validation:
        raise AssertionError(f"{graph_name}: validation DTI labels remain in graph")
    if forward & test:
        raise AssertionError(f"{graph_name}: test DTI labels remain in graph")


def _matrix_directory(dataset_dir: Path) -> Path:
    direct = dataset_dir / "mat_drug_protein.txt"
    return dataset_dir if direct.exists() else dataset_dir / "mat_data"


def _load_matrix(matrix_dir: Path, filename: str) -> np.ndarray:
    return np.loadtxt(matrix_dir / filename, dtype=np.int8)


def _identity(node_count: int) -> torch.Tensor:
    matrix = sp.eye(node_count, dtype=np.float32, format="coo")
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
    values = torch.from_numpy(matrix.data).float()
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def raw_graph(
    dataset: str,
    fold_index: int,
    task: str = "train",
    dataset_dir: Optional[Union[str, Path]] = None,
):
    """Build the two SGAN-DTI channels with training-positive DTI edges only."""
    del task  # The same training-only message-passing graph is used in all phases.
    root = (
        Path(dataset_dir).expanduser().resolve()
        if dataset_dir is not None
        else (Path(__file__).resolve().parents[1] / "datasets" / dataset).resolve()
    )
    loader = DataLoader(root, fold_index)
    matrix_dir = _matrix_directory(root)

    drug_drug = _load_matrix(matrix_dir, "mat_drug_drug.txt")
    drug_target = _load_matrix(matrix_dir, "mat_drug_protein.txt")
    drug_disease = _load_matrix(matrix_dir, "mat_drug_disease.txt")
    drug_side_effect = _load_matrix(matrix_dir, "mat_drug_se.txt")
    target_target = _load_matrix(matrix_dir, "mat_protein_protein.txt")
    target_disease = _load_matrix(matrix_dir, "mat_protein_disease.txt")

    def edges(matrix: np.ndarray):
        source, destination = matrix.nonzero()
        return torch.from_numpy(source), torch.from_numpy(destination)

    dd, dt, di, ds = map(edges, (drug_drug, drug_target, drug_disease, drug_side_effect))
    tt, ti = map(edges, (target_target, target_disease))
    graph_data = {
        (DRUG, "d_d", DRUG): dd,
        (DRUG, "d_p", TARGET): dt,
        (DRUG, "d_i", DISEASE): di,
        (DRUG, "d_s", SIDE_EFFECT): ds,
        (TARGET, "p_p", TARGET): tt,
        (TARGET, "p_i", DISEASE): ti,
        (TARGET, "p_d", DRUG): (dt[1], dt[0]),
        (DISEASE, "i_d", DRUG): (di[1], di[0]),
        (SIDE_EFFECT, "s_d", DRUG): (ds[1], ds[0]),
        (DISEASE, "i_p", TARGET): (ti[1], ti[0]),
    }
    graph = dgl.heterograph(
        graph_data,
        num_nodes_dict={
            DRUG: loader.node_counts["drug"],
            TARGET: loader.node_counts["target"],
            DISEASE: loader.node_counts["disease"],
            SIDE_EFFECT: loader.node_counts["side_effect"],
        },
    )

    # Convert the globally offset target ids in split files to graph-local ids.
    train = loader.train_pos.copy()
    validation = loader.valid_pos.copy()
    test = loader.test_pos.copy()
    for pairs in (train, validation, test):
        pairs[1] -= loader.drug_num

    drug_channel = dgl.edge_type_subgraph(
        graph, ["d_d", "d_p", "d_i", "d_s", "p_d", "i_d", "s_d"]
    )
    target_channel = dgl.edge_type_subgraph(
        graph, ["p_p", "p_d", "p_i", "d_p", "i_p"]
    )
    for name, channel in (("drug_channel", drug_channel), ("target_channel", target_channel)):
        remove_heldout_dti_edges(channel, validation, test)
        assert_training_only_dti(channel, train, validation, test, name)
        for node_type in channel.ntypes:
            channel.nodes[node_type].data["feat"] = _identity(channel.num_nodes(node_type))

    print(
        "EDGE_AUDIT "
        f"dataset={dataset} fold={fold_index} train={train.shape[1]} "
        "validation_in_graph=0 test_in_graph=0",
        flush=True,
    )
    return drug_channel, target_channel

