"""
Tabular Models Registry and Factory for Lumbar Spinal Stenosis Classification.
"""
from typing import Dict, Type
import torch.nn as nn

from models.tabular_models.tabnet_model import TabNetModel
from models.tabular_models.ft_transformer import FTTransformerModel
from models.tabular_models.mlp_model import TabularMLPModel
from models.tabular_models.deepfm_model import DeepFMModel
from models.tabular_models.cnn_mlp_model import CNNMLPModel

TABULAR_MODELS: Dict[str, Type[nn.Module]] = {
    'tabnet': TabNetModel,
    'ft_transformer': FTTransformerModel,
    'mlp': TabularMLPModel,
    'deepfm': DeepFMModel,
    'cnn_mlp': CNNMLPModel,
}

def get_tabular_model(
    model_name: str, 
    input_dim: int, 
    num_classes: int = 3
) -> nn.Module:
    """Instantiates the specified tabular classification model."""
    name = model_name.lower()
    if name not in TABULAR_MODELS:
        raise ValueError(f"Unknown tabular model '{name}'. Available: {list(TABULAR_MODELS.keys())}")
    
    cls = TABULAR_MODELS[name]
    return cls(input_dim=input_dim, num_classes=num_classes)
