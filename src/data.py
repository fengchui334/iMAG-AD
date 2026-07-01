from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_feature(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "gene" not in frame.columns:
        raise ValueError(f"{path} must contain a 'gene' column")
    feature_cols = [column for column in frame.columns if column != "gene"]
    if len(feature_cols) != 12:
        raise ValueError(f"{path} must contain exactly 12 feature columns")
    if frame["gene"].isna().any() or frame["gene"].duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate gene identifiers")

    frame["gene"] = frame["gene"].astype(str)
    frame[feature_cols] = frame[feature_cols].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(frame[feature_cols].to_numpy()).all():
        raise ValueError(f"{path} contains non-finite feature values")
    return frame


def load_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"gene", "label"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain 'gene' and 'label' columns")
    if frame["gene"].isna().any() or frame["gene"].duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate gene identifiers")

    frame["gene"] = frame["gene"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError(f"{path} labels must be 0 or 1")
    frame["label"] = frame["label"].astype(int)
    return frame[["gene", "label"]]


def align_feature_view(frame: pd.DataFrame, nodes: list) -> np.ndarray:
    present = set(frame["gene"])
    matrix = frame.set_index("gene").reindex(nodes).fillna(0.0)
    matrix["view_present"] = [float(gene in present) for gene in nodes]
    return matrix.to_numpy(dtype=np.float32)


def load_normalized_graph(
    path: Path, node_to_index: dict, device: torch.device
) -> torch.Tensor:
    frame = pd.read_csv(path)
    required = {"gene1", "gene2", "weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain gene1, gene2, and weight columns")
    if frame[["gene1", "gene2", "weight"]].isna().any().any():
        raise ValueError(f"{path} contains missing edge values")

    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if not np.isfinite(frame["weight"].to_numpy()).all():
        raise ValueError(f"{path} contains non-finite edge weights")
    frame["row"] = frame["gene1"].astype(str).map(node_to_index)
    frame["col"] = frame["gene2"].astype(str).map(node_to_index)
    frame = frame.dropna(subset=["row", "col"])
    frame["row"] = frame["row"].astype(np.int64)
    frame["col"] = frame["col"].astype(np.int64)
    frame = frame[frame["row"] != frame["col"]]

    left = np.minimum(frame["row"].to_numpy(), frame["col"].to_numpy())
    right = np.maximum(frame["row"].to_numpy(), frame["col"].to_numpy())
    undirected = pd.DataFrame(
        {
            "row": left,
            "col": right,
            "value": frame["weight"].to_numpy(dtype=np.float32),
        }
    )
    undirected = undirected.groupby(
        ["row", "col"], as_index=False
    )["value"].max()

    source = undirected["row"].to_numpy(dtype=np.int64)
    target = undirected["col"].to_numpy(dtype=np.int64)
    edge_values = undirected["value"].to_numpy(dtype=np.float32)
    node_count = len(node_to_index)
    diagonal = np.arange(node_count, dtype=np.int64)
    row = np.concatenate((source, target, diagonal))
    col = np.concatenate((target, source, diagonal))
    value = np.concatenate(
        (edge_values, edge_values, np.ones(node_count, dtype=np.float32))
    )

    # Symmetric normalization: D^(-1/2) (A + I) D^(-1/2).
    degree = np.zeros(node_count, dtype=np.float32)
    np.add.at(degree, row, value)
    inverse_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    value = value * inverse_sqrt_degree[row] * inverse_sqrt_degree[col]

    indices = torch.tensor(
        np.vstack((row, col)), dtype=torch.long, device=device
    )
    values = torch.tensor(value, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, (node_count, node_count), device=device
    ).coalesce()


def load_inputs(args: Namespace):
    device = torch.device(args.device)
    feature_paths = [
        Path(args.gene_feature),
        Path(args.isoform_feature),
        Path(args.ratio_feature),
    ]
    graph_paths = [
        Path(args.gene_graph),
        Path(args.isoform_graph),
        Path(args.ratio_graph),
    ]

    feature_frames = [load_feature(path) for path in feature_paths]
    nodes = sorted(
        set().union(*(set(frame["gene"]) for frame in feature_frames))
    )
    node_to_index = {gene: index for index, gene in enumerate(nodes)}
    features = [
        align_feature_view(frame, nodes) for frame in feature_frames
    ]
    graphs = [
        load_normalized_graph(path, node_to_index, device)
        for path in graph_paths
    ]

    labels = load_labels(Path(args.labels))
    labels = labels[labels["gene"].isin(node_to_index)].copy()
    label_indices = np.asarray(
        [node_to_index[gene] for gene in labels["gene"]], dtype=np.int64
    )
    label_values = labels["label"].to_numpy(dtype=np.int64)

    return (
        nodes,
        features,
        graphs,
        labels,
        label_indices,
        label_values,
    )
