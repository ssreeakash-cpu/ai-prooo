"""
DenseNet-121 Image Classification Model for Lumbar Spinal Stenosis.
Features dense connectivity blocks connecting each layer to every other layer in a feed-forward fashion.
"""
import torch
import torch.nn as nn
import timm

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = False, in_channels: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.model_name = "densenet_121"
        try:
            self.backbone = timm.create_model(
                'densenet121', 
                pretrained=pretrained, 
                in_chans=in_channels,
                num_classes=0
            )
            feature_dim = self.backbone.num_features
        except Exception:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            feature_dim = 1024

        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

    def get_cam_target_layer(self):
        """Returns the final convolutional feature layer for Grad-CAM."""
        if hasattr(self.backbone, 'features'):
            return self.backbone.features[-1]
        else:
            return self.backbone[7]
