"""
Clinical Inference Pipeline for Lumbar Spinal Stenosis (LSS).
Takes an unseen patient MRI scan (DICOM files or image), extracts key canal slices
at each level (L1/L2 to L5/S1), predicts 3-grade severity, and generates Grad-CAM visual heatmaps.
"""
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

from config.config import CONFIG
from data.slice_selector import CanalCropper, read_and_window_dicom
from models.image_models import get_image_model
from evaluation.gradcam import GradCAM

class LSSClinicalPredictor:
    """
    End-to-end clinical predictor for Lumbar Spinal Stenosis.
    """
    def __init__(self, model_name: str = "densenet_121", checkpoint_path: Optional[Path] = None):
        self.device = CONFIG.device
        self.model_name = model_name
        self.cropper = CanalCropper(target_size=CONFIG.image_size)

        # Load image classifier
        self.model = get_image_model(model_name, num_classes=CONFIG.num_classes).to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = CONFIG.checkpoints_dir / f"image_{model_name}.pt"
        
        if checkpoint_path.exists():
            print(f"Loading trained weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

        self.model.eval()

    def preprocess_image(self, img_input: Union[Path, str, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Loads and prepares MRI slice for network inference, returning:
        - normalized tensor for neural network
        - cropped canal region
        - full original spinal MRI slice
        """
        if isinstance(img_input, (str, Path)):
            path = Path(img_input)
            if path.suffix.lower() == '.dcm':
                norm_img = read_and_window_dicom(path)
            else:
                bgr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                norm_img = bgr.astype(np.float32) / 255.0
        else:
            norm_img = img_input.astype(np.float32)
            if norm_img.max() > 1.0:
                norm_img = norm_img / 255.0

        # Canal crop for model classification
        cropped = self.cropper.crop_canal_roi(norm_img)
        # 3-channel RGB replication
        rgb = np.stack([cropped] * 3, axis=-1).astype(np.float32)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_norm = (rgb - mean) / std
        tensor = torch.tensor(rgb_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, cropped, norm_img

    def annotate_full_spine_mri(self, full_img: np.ndarray, level: str, severity: str) -> np.ndarray:
        """
        Creates an annotated full spinal MRI view highlighting the targeted lumbar level.
        """
        h, w = full_img.shape[:2]
        full_uint8 = np.uint8(np.clip(full_img * 255.0, 0, 255))
        bgr = cv2.cvtColor(full_uint8, cv2.COLOR_GRAY2BGR)

        # Lumbar level anatomical vertical mappings (L1/L2 to L5/S1)
        level_y_fractions = {
            'L1/L2': 0.22,
            'L2/L3': 0.35,
            'L3/L4': 0.48,
            'L4/L5': 0.61,
            'L5/S1': 0.74
        }
        y_frac = level_y_fractions.get(level, 0.61)
        target_y = int(h * y_frac)
        target_x = int(w * 0.48)

        # Draw anatomical target bounding box around target level
        box_w = int(w * 0.42)
        box_h = int(h * 0.16)
        x1 = max(10, target_x - box_w // 2)
        x2 = min(w - 10, target_x + box_w // 2)
        y1 = max(10, target_y - box_h // 2)
        y2 = min(h - 10, target_y + box_h // 2)

        # Color coding: Green for Normal/Mild, Orange for Moderate, Crimson for Severe
        box_color = (40, 210, 80) if severity == 'Normal/Mild' else ((30, 160, 255) if severity == 'Moderate' else (50, 50, 240))
        
        # Target box & corner brackets
        cv2.rectangle(bgr, (x1, y1), (x2, y2), box_color, 2)
        bracket_len = 15
        cv2.line(bgr, (x1, y1), (x1 + bracket_len, y1), (255, 255, 255), 3)
        cv2.line(bgr, (x1, y1), (x1, y1 + bracket_len), (255, 255, 255), 3)
        cv2.line(bgr, (x2, y2), (x2 - bracket_len, y2), (255, 255, 255), 3)
        cv2.line(bgr, (x2, y2), (x2, y2 - bracket_len), (255, 255, 255), 3)

        # Label badge
        badge_text = f"TARGET: {level} CANAL"
        cv2.rectangle(bgr, (x1, max(0, y1 - 24)), (x1 + 175, y1), box_color, -1)
        cv2.putText(bgr, badge_text, (x1 + 6, max(12, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # PACS scan info banner
        cv2.putText(bgr, "Sagittal Lumbar Spine MRI", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(bgr, "L1 -> S1 View", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        return bgr

    def create_diagnostic_composite(
        self, 
        full_annotated: np.ndarray, 
        canal_zoom: np.ndarray, 
        cam_overlay: np.ndarray, 
        level: str, 
        severity: str, 
        confidence: float
    ) -> np.ndarray:
        """
        Creates a high-resolution 3-panel PACS diagnostic composite:
        [ 1. Full Spine View (L1-S1) ] | [ 2. Canal ROI Zoom ] | [ 3. Grad-CAM Verification Heatmap ]
        """
        panel_h = 420
        # Resize panels to standardized height
        p1 = cv2.resize(full_annotated, (int(full_annotated.shape[1] * (panel_h / full_annotated.shape[0])), panel_h))
        p2 = cv2.resize(canal_zoom, (panel_h, panel_h))
        p3 = cv2.resize(cam_overlay, (panel_h, panel_h))

        # Add title headers to each panel
        def add_panel_header(img, title, sub):
            hdr = np.zeros((45, img.shape[1], 3), dtype=np.uint8)
            hdr[:] = (25, 35, 48)
            cv2.putText(hdr, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(hdr, sub, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 180, 230), 1, cv2.LINE_AA)
            return np.vstack([hdr, img])

        p1_h = add_panel_header(p1, "1. Full Lumbar Spine MRI", "Complete L1-S1 Spinal Alignment")
        p2_h = add_panel_header(p2, f"2. {level} Canal ROI (Zoom)", "Dural Sac & Disc Margins")
        p3_h = add_panel_header(p3, "3. Grad-CAM Visual Proof", f"AI Attention ({severity}: {confidence*100:.1f}%)")

        # Separate with subtle vertical dividers
        divider = np.zeros((p1_h.shape[0], 4, 3), dtype=np.uint8)
        divider[:] = (50, 70, 90)

        composite = np.hstack([p1_h, divider, p2_h, divider, p3_h])
        return composite

    def predict_slice(
        self, 
        img_input: Union[Path, str, np.ndarray], 
        level: str = "L4/L5", 
        generate_cam: bool = True
    ) -> Dict:
        """
        Predicts severity for an MRI slice with full spinal views and Grad-CAM verification.
        """
        tensor, raw_slice, full_norm_img = self.preprocess_image(img_input)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            predicted_severity = CONFIG.idx_to_class[pred_idx]

        confidence = float(probs[pred_idx])

        # Full spinal annotated view
        full_annotated = self.annotate_full_spine_mri(full_norm_img, level, predicted_severity)

        # Canal Zoom view (grayscale to BGR)
        canal_uint8 = np.uint8(np.clip(raw_slice * 255.0, 0, 255))
        canal_zoom_bgr = cv2.cvtColor(canal_uint8, cv2.COLOR_GRAY2BGR)

        result = {
            'level': level,
            'predicted_severity': predicted_severity,
            'confidence': confidence,
            'probabilities': {
                'Normal/Mild': float(probs[0]),
                'Moderate': float(probs[1]),
                'Severe': float(probs[2])
            },
            'full_annotated_bgr': full_annotated,
            'canal_zoom_bgr': canal_zoom_bgr
        }

        # Grad-CAM Visual Heatmap
        if generate_cam:
            try:
                # Use spatial target layer if available
                target_layer = None
                if hasattr(self.model, 'get_cam_target_layer'):
                    target_layer = self.model.get_cam_target_layer()

                if target_layer is not None:
                    gradcam = GradCAM(self.model, target_layer=target_layer)
                    heatmap, _ = gradcam.generate_heatmap(tensor, target_class=pred_idx)
                    overlay = gradcam.overlay_on_image(raw_slice, heatmap, alpha=0.45)
                    result['heatmap'] = heatmap
                    result['overlay_bgr'] = overlay

                    # Create 3-Panel Composite
                    composite = self.create_diagnostic_composite(
                        full_annotated=full_annotated,
                        canal_zoom=canal_zoom_bgr,
                        cam_overlay=overlay,
                        level=level,
                        severity=predicted_severity,
                        confidence=confidence
                    )
                    result['composite_bgr'] = composite
            except Exception as e:
                result['cam_error'] = str(e)

        return result

    def predict_patient_study(self, study_dir: Path) -> List[Dict]:
        """
        Analyzes a complete patient scan volume across all 5 lower-back levels (L1/L2 to L5/S1).
        """
        levels = CONFIG.levels
        dicom_files = sorted(list(study_dir.glob("**/*.dcm")))
        if len(dicom_files) == 0:
            return []

        # Sample representative slices for each of the 5 levels
        n_slices = len(dicom_files)
        step = max(1, n_slices // len(levels))
        
        study_findings = []
        for i, lvl in enumerate(levels):
            slice_idx = min(n_slices - 1, i * step + step // 2)
            dcm_file = dicom_files[slice_idx]
            finding = self.predict_slice(dcm_file, level=lvl, generate_cam=True)
            finding['slice_file'] = str(dcm_file.name)
            study_findings.append(finding)

        return study_findings
