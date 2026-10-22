"""
TabNet Architecture for Tabular Spinal Stenosis Feature Classification.
Utilizes sequential attention with Sparsemax and Gated Linear Units (GLU) for feature selection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparsemax(nn.Module):
    """Sparsemax activation function for sparse feature selection."""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        # Sort input in descending order
        input_sorted, _ = torch.sort(input, descending=True, dim=self.dim)
        input_cumsum = torch.cumsum(input_sorted, dim=self.dim)
        
        # Determine k(z)
        rhos = torch.arange(1, input.size(self.dim) + 1, device=input.device)
        support = rhos * input_sorted > (input_cumsum - 1.0)
        # Find the number of non-zero elements
        k = torch.sum(support.long(), dim=self.dim, keepdim=True)
        tau = (torch.gather(input_cumsum, self.dim, k - 1) - 1.0) / k.float()
        
        output = torch.clamp(input - tau, min=0)
        return output

class GLU(nn.Module):
    """Gated Linear Unit: splits input into two halves and applies linear * sigmoid(linear)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        h = self.fc(x)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        return h1 * torch.sigmoid(h2)

class TabNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.glu1 = GLU(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.glu2 = GLU(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        h = self.bn1(self.glu1(x))
        h = self.bn2(self.glu2(h))
        return (h + x) * (0.5 ** 0.5) if x.shape[-1] == h.shape[-1] else h

class TabNetModel(nn.Module):
    """
    Self-contained PyTorch TabNet Model for Tabular Feature Classification.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 3, 
        n_d: int = 32, 
        n_a: int = 32, 
        n_steps: int = 3, 
        gamma: float = 1.3
    ):
        super().__init__()
        self.model_name = "tabnet"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.bn_in = nn.BatchNorm1d(input_dim)
        self.sparsemax = Sparsemax(dim=-1)

        # Attentive & Feature Transformers
        self.initial_transform = TabNetBlock(input_dim, n_d + n_a)
        
        self.att_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a, input_dim),
                nn.BatchNorm1d(input_dim)
            ) for _ in range(n_steps)
        ])

        self.feat_transformers = nn.ModuleList([
            TabNetBlock(input_dim, n_d + n_a) for _ in range(n_steps)
        ])

        self.final_classifier = nn.Linear(n_d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > 1:
            x = self.bn_in(x)
        
        prior = torch.ones_like(x)
        M_loss = 0.0
        out_agg = 0.0

        # Initial step
        h = self.initial_transform(x)
        d = h[:, :self.n_d]
        a = h[:, self.n_d:]

        for step in range(self.n_steps):
            # Attention mask
            mask_raw = self.att_transformers[step](a)
            mask = self.sparsemax(mask_raw * prior)
            prior = prior * (self.gamma - mask)

            # Masked features through feature transformer
            x_masked = x * mask
            h = self.feat_transformers[step](x_masked)
            d = F.relu(h[:, :self.n_d])
            a = h[:, self.n_d:]
            out_agg = out_agg + d

        logits = self.final_classifier(out_agg)
        return logits
