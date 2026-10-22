"""
Evaluation Metrics & Fair Scoring Module for Lumbar Spinal Stenosis.
Calculates Macro-F1, Precision, Recall, Confusion Matrix, and Adjacent-Grade Misclassification.
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

from config.config import CONFIG

def compute_fair_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict:
    """
    Computes rigorous clinical evaluation metrics:
    - Macro-F1 (unweighted average across classes to prevent dominance by Normal/Mild)
    - Severe Recall / Sensitivity (critical for clinical safety)
    - Adjacent-grade confusion rate
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class recalls
    per_class_recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    severe_recall = per_class_recalls[2] if len(per_class_recalls) > 2 else 0.0
    moderate_recall = per_class_recalls[1] if len(per_class_recalls) > 1 else 0.0
    normal_recall = per_class_recalls[0] if len(per_class_recalls) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Adjacent grade error: |true - pred| == 1
    # Extreme error: |true - pred| == 2 (e.g. Normal predicted as Severe or vice-versa)
    total_samples = len(y_true)
    diff = np.abs(y_true - y_pred)
    adjacent_error_rate = np.sum(diff == 1) / max(1, total_samples)
    extreme_error_rate = np.sum(diff == 2) / max(1, total_samples)

    metrics = {
        'model_name': model_name,
        'accuracy': float(acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'normal_recall': float(normal_recall),
        'moderate_recall': float(moderate_recall),
        'severe_recall': float(severe_recall),
        'adjacent_error_rate': float(adjacent_error_rate),
        'extreme_error_rate': float(extreme_error_rate),
        'confusion_matrix': cm.tolist()
    }

    return metrics

def plot_confusion_matrix(
    cm: np.ndarray, 
    model_name: str, 
    save_path: Optional[Path] = None, 
    class_names: Optional[List[str]] = None
):
    """Plots and saves an annotated confusion matrix heatmap."""
    if class_names is None:
        class_names = CONFIG.severity_classes

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names, 
        ax=ax,
        cbar=False
    )
    ax.set_title(f"Confusion Matrix: {model_name}\n(Spinal Stenosis Severity)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted Severity Grade", fontsize=10)
    ax.set_ylabel("True Expert Severity Grade", fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=200)
    plt.close()
