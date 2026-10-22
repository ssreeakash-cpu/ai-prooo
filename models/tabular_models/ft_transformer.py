"""
FT-Transformer (Feature Tokenizer Transformer) for Tabular Spinal Stenosis Prediction.
Tokenizes each tabular column into a vector representation and passes through Transformer layers.
"""
import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    """Tokenizes continuous tabular features into d-dimensional embedding vectors."""
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        # Individual weight vectors for each tabular feature
        self.weights = nn.Parameter(torch.randn(num_features, d_token))
        self.biases = nn.Parameter(torch.randn(num_features, d_token))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_features] -> [B, num_features, d_token]
        tokens = x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return tokens

class FTTransformerModel(nn.Module):
    """
    Feature Tokenizer Transformer for Tabular Classification.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 3, 
        d_token: int = 64, 
        n_heads: int = 4, 
        n_blocks: int = 3, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = "ft_transformer"
        self.tokenizer = FeatureTokenizer(input_dim, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_token, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Tokenize features
        tokens = self.tokenizer(x)  # [B, N, d_token]
        # Append [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, N+1, d_token]

        # Pass through Transformer blocks
        trans_out = self.transformer(tokens)
        cls_rep = self.norm(trans_out[:, 0])

        logits = self.head(cls_rep)
        return logits
