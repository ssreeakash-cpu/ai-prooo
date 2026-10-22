"""
CNN-MLP Hybrid Architecture for Tabular Spinal Stenosis Prediction.
Extracts localized coordinate and inter-level relational patterns via 1D Convolutions,
followed by a Multi-Layer Perceptron classification head.
"""
import torch
import torch.nn as nn

class CNNMLPModel(nn.Module):
    """
    1D CNN + MLP Hybrid Model for Tabular Features.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 3, 
        conv_channels: list = None, 
        mlp_dim: int = 64, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.model_name = "cnn_mlp"
        if conv_channels is None:
            conv_channels = [32, 64]

        # 1D Convolutional pipeline: treats [B, input_dim] as [B, 1, input_dim]
        c1, c2 = conv_channels
        self.conv1 = nn.Conv1d(1, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.act = nn.LeakyReLU(0.1)

        conv_out_dim = c2 * 4

        # MLP Pipeline
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim + input_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.BatchNorm1d(mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        x_seq = x.unsqueeze(1)  # [B, 1, input_dim]
        h_conv = self.act(self.bn1(self.conv1(x_seq)))
        h_conv = self.act(self.bn2(self.conv2(h_conv)))
        h_conv = self.pool(h_conv).flatten(1)  # [B, conv_out_dim]

        # Residual combination of raw tabular features + 1D CNN features
        combined = torch.cat([h_conv, x], dim=1)
        logits = self.mlp(combined)
        return logits
