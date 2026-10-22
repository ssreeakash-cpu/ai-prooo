"""
Central Configuration for Lumbar Spinal Stenosis (LSS) Detection & Classification System.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import torch

@dataclass
class LSSConfig:
    # Project paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data" / "dataset"
    raw_dicom_dir: Path = data_dir / "train_images"
    train_csv_path: Path = data_dir / "train.csv"
    series_csv_path: Path = data_dir / "train_series_descriptions.csv"
    coords_csv_path: Path = data_dir / "train_label_coordinates.csv"
    output_dir: Path = project_root / "outputs"
    checkpoints_dir: Path = output_dir / "checkpoints"
    heatmaps_dir: Path = output_dir / "heatmaps"
    reports_dir: Path = output_dir / "reports"

    # Spinal Anatomy Levels & Conditions
    levels: List[str] = field(default_factory=lambda: [
        'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1'
    ])
    
    conditions: List[str] = field(default_factory=lambda: [
        'Spinal Canal Stenosis',
        'Left Neural Foraminal Stenosis',
        'Right Neural Foraminal Stenosis',
        'Left Subarticular Stenosis',
        'Right Subarticular Stenosis'
    ])

    # Severity Grading Classes (RSNA 3-grade standard)
    severity_classes: List[str] = field(default_factory=lambda: [
        'Normal/Mild', 'Moderate', 'Severe'
    ])
    num_classes: int = 3
    class_to_idx: dict = field(default_factory=lambda: {
        'Normal/Mild': 0,
        'Moderate': 1,
        'Severe': 2
    })
    idx_to_class: dict = field(default_factory=lambda: {
        0: 'Normal/Mild',
        1: 'Moderate',
        2: 'Severe'
    })

    # Image Preprocessing & Cropping
    image_size: Tuple[int, int] = (224, 224)
    in_channels: int = 3  # RGB or replicated grayscale MRI
    crop_canal_margin: float = 0.25  # Margin around detected spinal canal ROI

    # Candidate Models
    image_models: List[str] = field(default_factory=lambda: [
        'efficientnet_b3',
        'densenet_121',
        'vision_transformer',
        'yolo_classifier',
        'unet_classifier'
    ])
    
    tabular_models: List[str] = field(default_factory=lambda: [
        'tabnet',
        'ft_transformer',
        'mlp',
        'deepfm',
        'cnn_mlp'
    ])

    # Training Hyperparameters
    batch_size: int = 16
    tabular_batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    focal_loss_gamma: float = 2.0
    use_focal_loss: bool = True
    num_workers: int = 0
    seed: int = 42

    # Hardware Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Create output directories if they don't exist
        for d in [self.output_dir, self.checkpoints_dir, self.heatmaps_dir, self.reports_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

# Global default config instance
CONFIG = LSSConfig()
