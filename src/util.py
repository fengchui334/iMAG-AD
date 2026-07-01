from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_from_training(
    matrix: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    train_matrix = matrix[train_indices]
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    matrix = (matrix - mean) / std
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(labels))
    selected = np.argsort(-scores)[:k]
    return float(labels[selected].sum() / k)


def calculate_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "Precision@20": precision_at_k(labels, scores, 20),
        "Precision@50": precision_at_k(labels, scores, 50),
        "Precision@100": precision_at_k(labels, scores, 100),
    }
