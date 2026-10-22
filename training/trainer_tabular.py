"""
Tabular Classification Trainer for Lumbar Spinal Stenosis Models.
Trains TabNet, FT-Transformer, MLP, DeepFM, and CNN-MLP on clinical & spatial features.
"""
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import CONFIG
from models.tabular_models import get_tabular_model
from training.loss import FocalLoss, compute_class_weights
from evaluation.metrics import compute_fair_metrics, plot_confusion_matrix

class TabularModelTrainer:
    def __init__(
        self, 
        model_name: str, 
        input_dim: int, 
        device: str = None, 
        epochs: int = None, 
        lr: float = 1e-3,
        use_focal: bool = True
    ):
        self.model_name = model_name
        self.input_dim = input_dim
        self.device = device or CONFIG.device
        self.epochs = epochs or (CONFIG.num_epochs + 2)
        self.lr = lr
        self.use_focal = use_focal
        self.model = get_tabular_model(model_name, input_dim=input_dim, num_classes=CONFIG.num_classes).to(self.device)

    def train_and_evaluate(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        class_weights: torch.Tensor = None
    ) -> Tuple[Dict, nn.Module]:
        """
        Trains the tabular model and calculates fair scoring metrics on validation data.
        """
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        if self.use_focal:
            criterion = FocalLoss(alpha=class_weights, gamma=CONFIG.focal_loss_gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=CONFIG.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_val_f1 = -1.0
        best_state = None

        print(f"\n--- Training Tabular Model: {self.model_name.upper()} on {self.device} ---")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0

            for X_b, y_b in train_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_b.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device)
                    outputs = self.model(X_b)
                    v_loss = criterion(outputs, y_b)
                    val_loss += v_loss.item() * X_b.size(0)

                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(y_b.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            epoch_metrics = compute_fair_metrics(np.array(val_targets), np.array(val_preds), model_name=self.model_name)
            val_f1 = epoch_metrics['macro_f1']
            scheduler.step(val_f1)

            print(f"Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f} | Severe Recall: {epoch_metrics['severe_recall']:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(self.device)
                outputs = self.model(X_b)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_b.numpy())

        final_metrics = compute_fair_metrics(np.array(all_targets), np.array(all_preds), model_name=self.model_name)

        # Save confusion matrix plot
        cm_path = CONFIG.reports_dir / f"cm_tabular_{self.model_name}.png"
        plot_confusion_matrix(np.array(final_metrics['confusion_matrix']), self.model_name, save_path=cm_path)

        # Save checkpoint
        ckpt_path = CONFIG.checkpoints_dir / f"tabular_{self.model_name}.pt"
        torch.save(self.model.state_dict(), ckpt_path)
        final_metrics['checkpoint_path'] = str(ckpt_path)

        return final_metrics, self.model
