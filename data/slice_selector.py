"""
Slice Selection and Spinal Canal ROI Cropping Module.
Targets specific lower-back levels (L1/L2 down to L5/S1) and selects/crops slices
displaying the spinal canal and thecal sac.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pydicom
import cv2
from pathlib import Path

from config.config import CONFIG

class SliceSelector:
    """
    Selects the most informative MRI slices for assessing spinal stenosis at each level (L1/L2 - L5/S1).
    """
    def __init__(self, coords_df: Optional[pd.DataFrame] = None):
        self.coords_df = coords_df

    def get_key_slices_for_series(self, series_dir: Path, series_desc: str) -> List[Path]:
        """
        Filters and ranks slices in a series. For Sagittal series, the central 20-40% 
        slices contain the median spinal canal and dural sac. For Axial series, slices are ordered 
        cranio-caudally from L1 down to S1.
        """
        slice_files = sorted(list(series_dir.glob("*.dcm")), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
        n_slices = len(slice_files)
        if n_slices == 0:
            return []

        if "Sagittal" in series_desc:
            # Mid-sagittal canal slices are concentrated around the volume center
            start = max(0, int(n_slices * 0.35))
            end = min(n_slices, int(n_slices * 0.65) + 1)
            selected = slice_files[start:end]
            return selected if len(selected) > 0 else slice_files
        else:
            # Axial series: evenly sample across the lumbar stack
            return slice_files

    def select_level_slices(
        self, 
        study_id: int, 
        series_id: int, 
        level: str, 
        condition: str = 'Spinal Canal Stenosis'
    ) -> Optional[int]:
        """
        Uses ground-truth coordinate annotations (if available) to locate the exact 
        instance_number representing the spinal level.
        """
        if self.coords_df is None or len(self.coords_df) == 0:
            return None

        matched = self.coords_df[
            (self.coords_df['study_id'] == study_id) &
            (self.coords_df['series_id'] == series_id) &
            (self.coords_df['level'] == level) &
            (self.coords_df['condition'] == condition)
        ]

        if not matched.empty:
            return int(matched.iloc[0]['instance_number'])
        return None

class CanalCropper:
    """
    Crops region-of-interest (ROI) tightly focused on the spinal canal,
    removing surrounding subcutaneous fat and non-diagnostic background.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224), margin: float = 0.25):
        self.target_size = target_size
        self.margin = margin

    def crop_canal_roi(
        self, 
        image: np.ndarray, 
        coord_x: Optional[float] = None, 
        coord_y: Optional[float] = None
    ) -> np.ndarray:
        """
        Crops spinal canal region. If coordinates (coord_x, coord_y) are provided, 
        centers crop around the anatomical landmark. Otherwise, uses an anatomical canal prior.
        """
        h, w = image.shape[:2]

        if coord_x is not None and coord_y is not None:
            cx, cy = int(coord_x), int(coord_y)
            crop_radius = int(min(h, w) * (0.20 + self.margin))
            x1 = max(0, cx - crop_radius)
            y1 = max(0, cy - crop_radius)
            x2 = min(w, cx + crop_radius)
            y2 = min(h, cy + crop_radius)
        else:
            # Anatomical prior: Lumbar spinal canal lies vertically in the central-posterior third
            x1 = int(w * 0.25)
            x2 = int(w * 0.75)
            y1 = int(h * 0.15)
            y2 = int(h * 0.85)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = image

        # Resize to standardized target size
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized

def read_and_window_dicom(
    dcm_path: Path, 
    window_center: Optional[float] = None, 
    window_width: Optional[float] = None
) -> np.ndarray:
    """
    Reads a DICOM file, applies RescaleSlope/Intercept and standard Hounsfield/MRI windowing, 
    and normalizes to float32 in [0, 1].
    """
    ds = pydicom.dcmread(str(dcm_path), force=True)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply slope & intercept if present
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    pixel_array = pixel_array * slope + intercept

    wc = window_center if window_center is not None else getattr(ds, 'WindowCenter', None)
    ww = window_width if window_width is not None else getattr(ds, 'WindowWidth', None)

    if isinstance(wc, (list, pydicom.multival.MultiValue)):
        wc = float(wc[0])
    if isinstance(ww, (list, pydicom.multival.MultiValue)):
        ww = float(ww[0])

    if wc is not None and ww is not None:
        min_val = wc - ww / 2.0
        max_val = wc + ww / 2.0
        pixel_array = np.clip(pixel_array, min_val, max_val)
        norm_img = (pixel_array - min_val) / (max_val - min_val + 1e-6)
    else:
        # Min-max normalization
        min_v = pixel_array.min()
        max_v = pixel_array.max()
        norm_img = (pixel_array - min_v) / (max_v - min_v + 1e-6)

    return np.clip(norm_img, 0.0, 1.0)
