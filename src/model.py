from __future__ import annotations

import torch
import torch.nn as nn

from layer import FeatureEncoder, GraphLayer


class IMAGAD(nn.Module):
    def __init__(
        self,
        input_dims: list,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_encoders = nn.ModuleList(
            [
                FeatureEncoder(input_dim, hidden_dim, dropout)
                for input_dim in input_dims
            ]
        )
        self.graph_layers = nn.ModuleList(
            [GraphLayer(hidden_dim, dropout) for _ in input_dims]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, features: list, graphs: list
    ) -> torch.Tensor:
        view_embeddings = [
            graph_layer(encoder(view_features), view_graph)
            for encoder, graph_layer, view_features, view_graph in zip(
                self.feature_encoders, self.graph_layers, features, graphs
            )
        ]
        return self.classifier(torch.cat(view_embeddings, dim=1))
