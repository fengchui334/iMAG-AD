from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from data import load_inputs
from model import IMAGAD
from util import calculate_metrics, set_seed, standardize_from_training


def fit(
    feature_matrices: list,
    labels_all_nodes: np.ndarray,
    train_indices: np.ndarray,
    graphs: list,
    args: Namespace,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    device = torch.device(args.device)
    features = [
        torch.tensor(matrix, dtype=torch.float32, device=device)
        for matrix in feature_matrices
    ]
    labels = torch.tensor(labels_all_nodes, dtype=torch.long, device=device)
    train_indices = torch.tensor(
        train_indices, dtype=torch.long, device=device
    )
    train_labels = labels[train_indices]

    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())
    class_weights = torch.tensor(
        [1.0, negatives / max(positives, 1)],
        dtype=torch.float32,
        device=device,
    )

    model = IMAGAD(
        [matrix.shape[1] for matrix in feature_matrices],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        logits = model(features, graphs)
        loss = F.cross_entropy(
            logits[train_indices], train_labels, weight=class_weights
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = torch.softmax(model(features, graphs), dim=1)[:, 1]
    return scores.cpu().numpy()


def cross_validate(args: Namespace) -> None:
    (
        nodes,
        raw_features,
        graphs,
        labels,
        label_indices,
        label_values,
    ) = load_inputs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    prediction_rows = []
    for experiment_seed in args.seeds:
        splitter = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=experiment_seed
        )
        folds = splitter.split(np.zeros(len(label_values)), label_values)

        for fold, (train_positions, test_positions) in enumerate(folds):
            if args.fold is not None and fold != args.fold:
                continue
            train_indices = label_indices[train_positions]
            test_indices = label_indices[test_positions]
            labels_all_nodes = np.zeros(len(nodes), dtype=np.int64)
            labels_all_nodes[label_indices] = label_values
            # Fit scaling on training genes only to avoid fold leakage.
            features = [
                standardize_from_training(matrix, train_indices)
                for matrix in raw_features
            ]
            scores = fit(
                features,
                labels_all_nodes,
                train_indices,
                graphs,
                args,
                seed=experiment_seed + fold,
            )

            test_labels = label_values[test_positions]
            test_scores = scores[test_indices]
            metrics = calculate_metrics(test_labels, test_scores)
            metric_rows.append(
                {"seed": experiment_seed, "fold": fold, **metrics}
            )
            prediction_rows.extend(
                {
                    "seed": experiment_seed,
                    "fold": fold,
                    "gene": gene,
                    "y_true": int(label),
                    "score": float(score),
                }
                for gene, label, score in zip(
                    labels.iloc[test_positions]["gene"],
                    test_labels,
                    test_scores,
                )
            )
            print(
                f"seed={experiment_seed} fold={fold} "
                f"AUPRC={metrics['AUPRC']:.4f} "
                f"AUROC={metrics['AUROC']:.4f}"
            )

    metric_columns = [
        "AUROC",
        "AUPRC",
        "Precision@20",
        "Precision@50",
        "Precision@100",
    ]
    fold_metrics = pd.DataFrame(metric_rows)
    seed_summary = (
        fold_metrics.groupby("seed", as_index=False)[metric_columns].mean()
    )
    summary = {
        "n_seeds": len(args.seeds),
        "n_folds": int(fold_metrics["fold"].nunique()),
    }
    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(seed_summary[metric].mean())
        summary[f"{metric}_std"] = (
            float(seed_summary[metric].std(ddof=1))
            if len(seed_summary) > 1
            else 0.0
        )

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "oof_predictions.csv", index=False
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def rank_candidates(args: Namespace) -> None:
    (
        nodes,
        raw_features,
        graphs,
        labels,
        label_indices,
        label_values,
    ) = load_inputs(args)
    labels_all_nodes = np.zeros(len(nodes), dtype=np.int64)
    labels_all_nodes[label_indices] = label_values
    # Final ranking uses all labels; unlabeled genes remain graph nodes.
    features = [
        standardize_from_training(matrix, label_indices)
        for matrix in raw_features
    ]

    score_runs = [
        fit(
            features,
            labels_all_nodes,
            label_indices,
            graphs,
            args,
            seed=seed,
        )
        for seed in args.seeds
    ]
    scores = np.vstack(score_runs)
    labeled_genes = set(labels["gene"])

    ranking = pd.DataFrame(
        {
            "gene": nodes,
            "imag_score": scores.mean(axis=0),
            "imag_score_std": scores.std(axis=0),
            "is_labeled": [gene in labeled_genes for gene in nodes],
            "label": [
                int(labels_all_nodes[index]) if gene in labeled_genes else -1
                for index, gene in enumerate(nodes)
            ],
        }
    )
    ranking = ranking.sort_values(
        "imag_score", ascending=False
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    candidates = ranking[~ranking["is_labeled"]].copy().reset_index(drop=True)
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "imag_full_ranking.csv", index=False)
    candidates.to_csv(output_dir / "imag_candidate_ranking.csv", index=False)
    print(f"ranked {len(candidates)} unlabeled candidate genes")
