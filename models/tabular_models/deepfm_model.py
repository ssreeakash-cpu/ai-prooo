"""
DeepFM (Deep Factorization Machine) Model for Tabular Spinal Stenosis Classification.
Combines 1st-order linear interactions, 2nd-order Factorization Machine pairwise interactions,
and higher-order non-linear interactions via a Deep Neural Network.
"""
import torch
import torch.nn as nn

class DeepFMModel(nn.Module):
    """
    DeepFM: FM component + Deep Neural Network component.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 3, 
        embedding_dim: int = 16, 
        hidden_dims: list = None, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.model_name = "deepfm"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # 1. FM First-Order Linear Component
        self.linear = nn.Linear(input_dim, num_classes, bias=True)

        # 2. FM Second-Order Feature Interaction Embeddings
        # Each input feature is projected to a k-dimensional embedding
        self.fm_embeddings = nn.Parameter(torch.randn(input_dim, embedding_dim))
        nn.init.xavier_uniform_(self.fm_embeddings)

        # 3. Deep Component
        deep_in_dim = input_dim * embedding_dim
        layers = []
        in_d = deep_in_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.BatchNorm1d(h_d),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_d = h_d
        layers.append(nn.Linear(in_d, num_classes))
        self.deep_network = nn.Sequential(*layers)

        # Second-order projection to num_classes
        self.fm_proj = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1st-order linear output: [B, num_classes]
        y_first_order = self.linear(x)

        # 2nd-order FM interaction:
        # v_i * x_i: [B, input_dim, embedding_dim]
        vx = x.unsqueeze(-1) * self.fm_embeddings.unsqueeze(0)
        # Sum then square
        sum_vx = torch.sum(vx, dim=1)  # [B, embedding_dim]
        sum_vx_squared = torch.pow(sum_vx, 2)
        # Square then sum
        squared_vx = torch.pow(vx, 2)
        sum_squared_vx = torch.sum(squared_vx, dim=1)  # [B, embedding_dim]
        # FM 2nd-order vector: 0.5 * ( (sum)^2 - sum(squares) )
        fm_vector = 0.5 * (sum_vx_squared - sum_squared_vx)
        y_second_order = self.fm_proj(fm_vector)  # [B, num_classes]

        # Deep component
        deep_in = vx.view(B, -1)  # Flatten embeddings: [B, input_dim * embedding_dim]
        y_deep = self.deep_network(deep_in)  # [B, num_classes]

        # Combined logits
        logits = y_first_order + y_second_order + y_deep
        return logits
