"""
U-Net Encoder-Decoder Classification Model for Lumbar Spinal Stenosis.
Combines contraction encoder stages with skip-connected decoder layers,
extracting both fine-grained canal boundary details and high-level context.
"""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNetClassifier(nn.Module):
    """
    U-Net Classifier with skip-connected encoder-decoder feature representation.
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.model_name = "unet_classifier"
        bf = base_filters

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 4, bf * 8))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 8, bf * 16))

        # Expansive Path (Decoder with skip connections)
        self.up1 = nn.ConvTranspose2d(bf * 16, bf * 8, 2, stride=2)
        self.dec1 = DoubleConv(bf * 16, bf * 8)

        self.up2 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.dec2 = DoubleConv(bf * 8, bf * 4)

        # Multi-scale feature pooling head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Concatenate bottleneck (bf*16) and decoder (bf*4)
        combined_dim = (bf * 16) + (bf * 4)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bottleneck(x4)

        # Decoder
        d1 = self.up1(b)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.dec2(d2)

        # Aggregate multi-scale representations
        feat_b = self.pool(b).flatten(1)
        feat_d = self.pool(d2).flatten(1)
        feat = torch.cat([feat_b, feat_d], dim=1)

        logits = self.fc(feat)
        return logits

    def get_cam_target_layer(self):
        """Returns the bottleneck double-conv layer for Grad-CAM."""
        return self.bottleneck[1].net[3]
