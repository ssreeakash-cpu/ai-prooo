"""
Deep Multi-Layer Perceptron (MLP) with Residual Connections for Tabular Feature Classification.
"""
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class TabularMLPModel(nn.Module):
    """
    Modern Tabular MLP with Residual Skip Connections.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 3, 
        hidden_dim: int = 128, 
        num_res_blocks: int = 2, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.model_name = "mlp"
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)
        ])

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)
        return logits
