"""
Imbalance-Aware Loss Functions for Lumbar Spinal Stenosis.
Includes Class-Weighted Cross Entropy and Multi-Class Focal Loss to ensure rare Severe cases
are not overlooked by the model.
"""
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_class_weights(labels: Union[np.ndarray, torch.Tensor], num_classes: int = 3) -> torch.Tensor:
    """
    Computes inverse frequency class weights to counteract class imbalance.
    Weights are inversely proportional to class frequencies.
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)

    counts = np.bincount(labels_np, minlength=num_classes)
    total_samples = len(labels_np)
    
    # Smooth zero occurrences
    counts = np.maximum(counts, 1)
    # Balanced weighting formula: total / (num_classes * count)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    
    # Normalize weights so that mean weight is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples (frequent Normal/Mild cases) and focuses gradient on hard, rare cases (Severe).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, num_classes], targets: [B]
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (p * target_one_hot).sum(dim=-1)
        log_p_t = (log_p * target_one_hot).sum(dim=-1)

        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = -focal_factor * log_p_t

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = (self.alpha * target_one_hot).sum(dim=-1)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
