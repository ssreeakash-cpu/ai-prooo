"""
PyTorch Datasets for Lumbar Spinal Stenosis:
- LSSImageDataset: Loads DICOM slices with canal cropping and preprocessing
- LSSTabularDataset: Structured tabular features from RSNA CSV files
"""
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config.config import CONFIG
from data.slice_selector import CanalCropper, SliceSelector, read_and_window_dicom

class LSSImageDataset(Dataset):
    """
    Dataset for medical MRI slice classification with canal-targeted cropping.
    """
    def __init__(
        self, 
        df_samples: pd.DataFrame, 
        raw_images_dir: Path, 
        transform: Optional[transforms.Compose] = None,
        crop_canal: bool = True
    ):
        self.df = df_samples.reset_index(drop=True)
        self.raw_images_dir = raw_images_dir
        self.cropper = CanalCropper(target_size=CONFIG.image_size) if crop_canal else None
        
        # Default medical image normalization
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        row = self.df.iloc[idx]
        study_id = int(row['study_id'])
        series_id = int(row['series_id'])
        instance_num = int(row['instance_number'])
        label_val = int(row['target_label'])

        dcm_path = self.raw_images_dir / str(study_id) / str(series_id) / f"{instance_num}.dcm"

        if dcm_path.exists():
            norm_img = read_and_window_dicom(dcm_path)
            coord_x = row.get('x', None)
            coord_y = row.get('y', None)
            if pd.isna(coord_x) or pd.isna(coord_y):
                coord_x, coord_y = None, None

            if self.cropper is not None:
                img_cropped = self.cropper.crop_canal_roi(norm_img, coord_x=coord_x, coord_y=coord_y)
            else:
                img_cropped = cv2.resize(norm_img, CONFIG.image_size)
        else:
            # Fallback placeholder if missing
            img_cropped = np.zeros(CONFIG.image_size, dtype=np.float32)

        # Replicate 1-channel grayscale to 3 channels for backbone compatibility
        img_rgb = np.stack([img_cropped] * 3, axis=-1).astype(np.float32)
        tensor_img = self.transform(img_rgb)

        meta = {
            'study_id': study_id,
            'series_id': series_id,
            'instance_number': instance_num,
            'level': str(row.get('level', 'L4/L5')),
            'condition': str(row.get('condition', 'Spinal Canal Stenosis')),
            'label_name': CONFIG.idx_to_class.get(label_val, 'Unknown')
        }

        return tensor_img, label_val, meta

class LSSTabularDataset(Dataset):
    """
    Structured tabular dataset representing clinical metadata, scan geometry,
    relative level coordinates, series descriptions, and patient demographics.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.X[idx], self.y[idx].item()

def prepare_lss_datasets(
    data_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads RSNA CSVs, aligns coordinate annotations and severity labels,
    and returns processed tabular sample frames ready for train/val/test splitting.
    """
    if data_dir is None:
        data_dir = CONFIG.data_dir

    train_csv = data_dir / "train.csv"
    series_csv = data_dir / "train_series_descriptions.csv"
    coords_csv = data_dir / "train_label_coordinates.csv"

    df_train = pd.read_csv(train_csv)
    df_series = pd.read_csv(series_csv)
    df_coords = pd.read_csv(coords_csv)

    # Melt train.csv to long format: (study_id, condition_level, severity)
    severity_map = CONFIG.class_to_idx
    id_vars = ['study_id']
    value_vars = [col for col in df_train.columns if col != 'study_id']
    
    df_melt = pd.melt(df_train, id_vars=id_vars, value_vars=value_vars, var_name='target_col', value_name='severity')
    df_melt['target_label'] = df_melt['severity'].map(severity_map).fillna(0).astype(int)

    # Parse condition and level from target_col
    def parse_condition_level(col: str):
        for lvl in ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']:
            if col.endswith(lvl):
                cond = col[:-len(lvl)-1].replace('_', ' ').title()
                level_disp = lvl.replace('_', '/').upper()
                return cond, level_disp
        return 'Spinal Canal Stenosis', 'L4/L5'

    parsed = df_melt['target_col'].apply(parse_condition_level)
    df_melt['condition'] = [p[0] for p in parsed]
    df_melt['level'] = [p[1] for p in parsed]

    # Merge with coordinates and series descriptions
    # Harmonize condition names
    cond_harmonize = {
        'Spinal Canal Stenosis': 'Spinal Canal Stenosis',
        'Left Neural Foraminal Stenosis': 'Left Neural Foraminal Stenosis',
        'Right Neural Foraminal Stenosis': 'Right Neural Foraminal Stenosis',
        'Left Subarticular Stenosis': 'Left Subarticular Stenosis',
        'Right Subarticular Stenosis': 'Right Subarticular Stenosis'
    }
    df_coords['condition'] = df_coords['condition'].map(lambda x: cond_harmonize.get(x, x))

    # Merge coordinates with melted labels
    df_merged = pd.merge(
        df_coords, 
        df_melt, 
        on=['study_id', 'condition', 'level'], 
        how='inner'
    )

    # Merge series descriptions
    df_final = pd.merge(df_merged, df_series, on=['study_id', 'series_id'], how='left')

    # Build structured tabular features
    # Numerical: x, y, instance_number
    # Categorical: series_description, level, condition
    level_dummies = pd.get_dummies(df_final['level'], prefix='lvl', drop_first=False)
    series_dummies = pd.get_dummies(df_final['series_description'], prefix='series', drop_first=False)
    cond_dummies = pd.get_dummies(df_final['condition'], prefix='cond', drop_first=False)

    # Normalized spatial features
    df_final['norm_x'] = df_final['x'] / 224.0
    df_final['norm_y'] = df_final['y'] / 224.0
    df_final['norm_slice'] = df_final['instance_number'] / 25.0

    tabular_feature_cols = ['norm_x', 'norm_y', 'norm_slice'] + list(level_dummies.columns) + list(series_dummies.columns) + list(cond_dummies.columns)
    
    df_tab_features = pd.concat([df_final[['norm_x', 'norm_y', 'norm_slice']], level_dummies, series_dummies, cond_dummies], axis=1).astype(float)
    
    for c in tabular_feature_cols:
        df_final[c] = df_tab_features[c]

    return df_final, df_train, df_series, tabular_feature_cols
