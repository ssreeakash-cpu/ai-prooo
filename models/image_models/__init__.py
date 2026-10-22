"""
Image Models Registry and Factory for Lumbar Spinal Stenosis Classification.
"""
from typing import Dict, Type
import torch.nn as nn

from models.image_models.efficientnet_b3 import EfficientNetB3Classifier
from models.image_models.densenet_121 import DenseNet121Classifier
from models.image_models.vision_transformer import VisionTransformerClassifier
from models.image_models.yolo_classifier import YOLOClassifier
from models.image_models.unet_classifier import UNetClassifier

IMAGE_MODELS: Dict[str, Type[nn.Module]] = {
    'efficientnet_b3': EfficientNetB3Classifier,
    'densenet_121': DenseNet121Classifier,
    'vision_transformer': VisionTransformerClassifier,
    'yolo_classifier': YOLOClassifier,
    'unet_classifier': UNetClassifier,
}

def get_image_model(
    model_name: str, 
    num_classes: int = 3, 
    pretrained: bool = False, 
    in_channels: int = 3
) -> nn.Module:
    """Instantiates the specified image classification model."""
    name = model_name.lower()
    if name not in IMAGE_MODELS:
        raise ValueError(f"Unknown image model '{name}'. Available: {list(IMAGE_MODELS.keys())}")
    
    cls = IMAGE_MODELS[name]
    if name in ['efficientnet_b3', 'densenet_121', 'vision_transformer']:
        return cls(num_classes=num_classes, pretrained=pretrained, in_channels=in_channels)
    else:
        return cls(num_classes=num_classes, in_channels=in_channels)
