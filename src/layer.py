from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class GraphLayer(nn.Module):
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.dropout = dropout
        self.graph_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, hidden: torch.Tensor, graph: torch.Tensor
    ) -> torch.Tensor:
        # The two sparse multiplications implement two-hop propagation.
        hidden = torch.sparse.mm(graph, hidden)
        hidden = F.relu(self.graph_linear(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return torch.sparse.mm(graph, hidden)
