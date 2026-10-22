"""
Vision Transformer (ViT) Image Classification Model for Lumbar Spinal Stenosis.
Transforms 16x16 image patches through Multi-Head Self-Attention (MHSA) encoder blocks.
"""
import math
import torch
import torch.nn as nn
import timm

class VisionTransformerClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int = 3, 
        pretrained: bool = False, 
        in_channels: int = 3, 
        img_size: int = 224, 
        patch_size: int = 16, 
        embed_dim: int = 384, 
        depth: int = 6, 
        num_heads: int = 6
    ):
        super().__init__()
        self.model_name = "vision_transformer"
        try:
            self.backbone = timm.create_model(
                'vit_tiny_patch16_224', 
                pretrained=pretrained, 
                in_chans=in_channels,
                num_classes=0
            )
            feature_dim = self.backbone.num_features
            self.use_timm = True
        except Exception:
            self.use_timm = False
            # Standalone Native Vision Transformer Implementation
            self.patch_size = patch_size
            self.num_patches = (img_size // patch_size) ** 2
            self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4, 
                dropout=0.1, 
                activation='gelu', 
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.norm = nn.LayerNorm(embed_dim)
            feature_dim = embed_dim

        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_timm:
            features = self.backbone(x)
        else:
            B = x.shape[0]
            # [B, in_channels, H, W] -> [B, embed_dim, H/P, W/P] -> [B, num_patches, embed_dim]
            patches = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_tokens = torch.cat((cls_tokens, patches), dim=1) + self.pos_embed
            trans_out = self.transformer(x_tokens)
            features = self.norm(trans_out[:, 0])

        logits = self.head(features)
        return logits

    def get_cam_target_layer(self):
        """Returns target layer for transformer attention map extraction."""
        if self.use_timm and hasattr(self.backbone, 'blocks'):
            return self.backbone.blocks[-1].norm1
        elif not self.use_timm:
            return self.transformer.layers[-1].norm1
        return None
