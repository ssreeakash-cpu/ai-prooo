"""
Multimodal Fusion Model for Lumbar Spinal Stenosis Assessment.
Fuses deep MRI visual representations from the top image model with structured
clinical and spatial features from the top tabular model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalLSSFusion(nn.Module):
    """
    Multimodal Fusion architecture combining image encoder logits/embeddings with tabular encoder.
    """
    def __init__(
        self, 
        image_model: nn.Module, 
        tabular_model: nn.Module, 
        num_classes: int = 3, 
        fusion_dim: int = 64
    ):
        super().__init__()
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.num_classes = num_classes

        # Joint fusion classifier head
        self.fusion_fc = nn.Sequential(
            nn.Linear(num_classes * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        # Get logits from both modalities
        img_logits = self.image_model(img)
        tab_logits = self.tabular_model(tab)

        # Concatenate modal representations
        combined = torch.cat([img_logits, tab_logits], dim=1)
        fused_logits = self.fusion_fc(combined)
        return fused_logits
