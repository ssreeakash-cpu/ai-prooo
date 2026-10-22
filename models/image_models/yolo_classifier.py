"""
YOLO Image Classification Model (YOLOv8 Architecture) for Lumbar Spinal Stenosis.
Implements Conv-BN-SiLU, C2f Cross Stage Partial bottlenecks, and SPPF (Spatial Pyramid Pooling Fast).
"""
import torch
import torch.nn as nn

class ConvModule(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation module."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard YOLO bottleneck with residual connection."""
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvModule(c1, c_, 3, 1)
        self.cv2 = ConvModule(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster implementation of CSP Bottleneck with 2 convolutions (C2f)."""
    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = ConvModule(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvModule((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for multi-scale receptive field aggregation."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvModule(c1, c_, 1, 1)
        self.cv2 = ConvModule(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class YOLOClassifier(nn.Module):
    """
    YOLOv8-based Classification Model.
    Processes MRI scans through CSP stages and SPPF for spinal canal feature extraction.
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3, width_mult: float = 0.5):
        super().__init__()
        self.model_name = "yolo_classifier"
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)

        self.stem = ConvModule(in_channels, c1, 3, 2)  # 224 -> 112
        self.stage1 = nn.Sequential(
            ConvModule(c1, c2, 3, 2),  # 112 -> 56
            C2f(c2, c2, n=1, shortcut=True)
        )
        self.stage2 = nn.Sequential(
            ConvModule(c2, c3, 3, 2),  # 56 -> 28
            C2f(c3, c3, n=2, shortcut=True)
        )
        self.stage3 = nn.Sequential(
            ConvModule(c3, c4, 3, 2),  # 28 -> 14
            C2f(c4, c4, n=2, shortcut=True)
        )
        self.sppf = SPPF(c4, c4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.2)
        self.linear = nn.Linear(c4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.sppf(x)
        features = self.pool(x).flatten(1)
        features = self.drop(features)
        logits = self.linear(features)
        return logits

    def get_cam_target_layer(self):
        """Returns the SPPF layer for Grad-CAM activation visualization."""
        return self.sppf.cv2.conv
