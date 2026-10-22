"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Visual Proof.
Generates color-coded heatmaps over MRI slices to verify whether the AI model focuses
specifically on the lumbar spinal canal, dural sac, and ligamentum flavum.
"""
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

from config.config import CONFIG

class GradCAM:
    """
    Computes Grad-CAM saliency maps for convolutional neural networks.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None and hasattr(model, 'get_cam_target_layer'):
            target_layer = model.get_cam_target_layer()

        if target_layer is None:
            raise ValueError(f"Target layer for Grad-CAM could not be resolved on {type(model).__name__}")

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generates 2D normalized Grad-CAM heatmap in [0, 1].
        """
        self.model.zero_grad()
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Check if activations are 4D (CNN: [B, C, H, W]) or 3D (Transformer: [B, N, D])
        if self.activations.dim() == 4:
            # Global average pooling of gradients: [C]
            weights = torch.mean(self.gradients[0], dim=(1, 2))  # [C]
            cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=input_tensor.device)
            for i, w in enumerate(weights):
                cam += w * self.activations[0, i]
            cam = torch.relu(cam)
            cam_np = cam.cpu().numpy()
        else:
            # Transformer tokens: [B, N, D]
            tokens = self.activations[0]
            grads = self.gradients[0]
            if tokens.shape[0] > 196:
                tokens = tokens[1:]
                grads = grads[1:]
            weights = torch.mean(grads, dim=1)  # [N]
            token_cam = torch.relu(torch.sum(tokens * grads, dim=1))
            n = token_cam.shape[0]
            grid = int(np.sqrt(n))
            if grid * grid == n:
                cam_np = token_cam.view(grid, grid).cpu().numpy()
            else:
                cam_np = np.resize(token_cam.cpu().numpy(), (14, 14))
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-6:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class

    def overlay_on_image(
        self, 
        original_image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.5, 
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlays the color-coded heatmap onto the original grayscale MRI slice.
        """
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        if original_image.ndim == 2:
            orig_rgb = cv2.cvtColor(np.uint8(255 * original_image), cv2.COLOR_GRAY2BGR)
        elif original_image.shape[2] == 1:
            orig_rgb = cv2.cvtColor(np.uint8(255 * original_image), cv2.COLOR_GRAY2BGR)
        else:
            orig_rgb = np.uint8(255 * original_image) if original_image.max() <= 1.0 else original_image.astype(np.uint8)

        overlay = cv2.addWeighted(orig_rgb, 1.0 - alpha, color_heatmap, alpha, 0)
        return overlay

def generate_and_save_visual_proof(
    model: nn.Module,
    img_tensor: torch.Tensor,
    raw_slice: np.ndarray,
    level: str,
    true_label: str,
    pred_label: str,
    save_path: Path
):
    """
    Creates and saves a side-by-side diagnostic visualization showing:
    1. Raw Patient MRI scan
    2. Grad-CAM visual attention heatmap highlighting the spinal canal
    3. Diagnostic overlay with severity grade prediction
    """
    grad_cam = GradCAM(model)
    heatmap, pred_idx = grad_cam.generate_heatmap(img_tensor)
    overlay = grad_cam.overlay_on_image(raw_slice, heatmap, alpha=0.45)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Raw MRI
    axes[0].imshow(raw_slice, cmap='gray')
    axes[0].set_title(f"Patient Lumbar MRI ({level})\nTrue Grade: {true_label}", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. Grad-CAM Heatmap
    im_cam = axes[1].imshow(heatmap, cmap='turbo')
    axes[1].set_title(f"Grad-CAM Canal Saliency Map\n(Target: {pred_label})", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im_cam, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Diagnostic Overlay
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"AI Anatomical Verification\nPredicted: {pred_label}", fontsize=11, fontweight='bold', color='darkgreen' if true_label == pred_label else 'darkred')
    axes[2].axis('off')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    return save_path
