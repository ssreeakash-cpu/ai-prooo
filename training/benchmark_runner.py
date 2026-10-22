"""
Master Benchmark Runner for Lumbar Spinal Stenosis AI System.
Executes end-to-end:
1. Patient-level split ("No Cheating")
2. Benchmarks all 5 Image Models -> Selects Top 3 -> Compares Top 3 -> Elects Champion Image Model
3. Benchmarks all 5 Tabular Models -> Selects Top 3 -> Compares Top 3 -> Elects Champion Tabular Model
4. Trains Multimodal Fusion (Top Image + Top Tabular)
5. Generates Grad-CAM visual proof heatmaps
"""
from typing import Dict, List, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.config import CONFIG
from data.dataset import LSSImageDataset, LSSTabularDataset, prepare_lss_datasets
from data.patient_splitter import PatientDataSplitter
from training.loss import compute_class_weights
from training.trainer_image import ImageModelTrainer
from training.trainer_tabular import TabularModelTrainer
from models.multimodal_fusion import MultimodalLSSFusion
from evaluation.metrics import compute_fair_metrics, plot_confusion_matrix
from evaluation.model_comparator import ModelComparator
from evaluation.gradcam import generate_and_save_visual_proof

def run_lss_benchmark(data_dir: Path = None, num_epochs: int = None) -> Dict:
    """
    Executes the comprehensive benchmark suite across all models.
    """
    if data_dir is None:
        data_dir = CONFIG.data_dir
    epochs = num_epochs or CONFIG.num_epochs

    print("=" * 80)
    print("  LUMBAR SPINAL STENOSIS (LSS) MULTI-MODEL BENCHMARK & COMPARISON PIPELINE")
    print("=" * 80)

    # 1. Prepare Datasets & Patient-Level Splitting ("No Cheating")
    print("\n[Step 1/5] Loading RSNA Dataset & Partitioning Patients ('No Cheating')...")
    df_samples, df_train, df_series, feature_cols = prepare_lss_datasets(data_dir)

    splitter = PatientDataSplitter(seed=CONFIG.seed)
    df_train_split, df_val_split, df_test_split = splitter.split_patients(df_samples, patient_col='study_id')

    # Compute class weights from training labels to handle class imbalance
    train_labels = df_train_split['target_label'].values
    class_weights = compute_class_weights(train_labels, num_classes=CONFIG.num_classes)
    print(f"\nHandling Imbalance: Computed Class Weights [Normal/Mild, Moderate, Severe]:")
    print(f"  {class_weights.numpy().round(3)}")

    # Image DataLoaders
    train_img_ds = LSSImageDataset(df_train_split, raw_images_dir=data_dir / "train_images")
    val_img_ds = LSSImageDataset(df_val_split, raw_images_dir=data_dir / "train_images")
    test_img_ds = LSSImageDataset(df_test_split, raw_images_dir=data_dir / "train_images")

    train_img_loader = DataLoader(train_img_ds, batch_size=CONFIG.batch_size, shuffle=True)
    val_img_loader = DataLoader(val_img_ds, batch_size=CONFIG.batch_size, shuffle=False)
    test_img_loader = DataLoader(test_img_ds, batch_size=CONFIG.batch_size, shuffle=False)

    # Tabular DataLoaders
    X_train = df_train_split[feature_cols].values.astype(np.float32)
    y_train = df_train_split['target_label'].values.astype(np.int64)
    X_val = df_val_split[feature_cols].values.astype(np.float32)
    y_val = df_val_split['target_label'].values.astype(np.int64)
    X_test = df_test_split[feature_cols].values.astype(np.float32)
    y_test = df_test_split['target_label'].values.astype(np.int64)

    train_tab_ds = LSSTabularDataset(X_train, y_train, feature_cols)
    val_tab_ds = LSSTabularDataset(X_val, y_val, feature_cols)
    test_tab_ds = LSSTabularDataset(X_test, y_test, feature_cols)

    train_tab_loader = DataLoader(train_tab_ds, batch_size=CONFIG.tabular_batch_size, shuffle=True)
    val_tab_loader = DataLoader(val_tab_ds, batch_size=CONFIG.tabular_batch_size, shuffle=False)
    test_tab_loader = DataLoader(test_tab_ds, batch_size=CONFIG.tabular_batch_size, shuffle=False)

    comparator = ModelComparator(output_dir=CONFIG.reports_dir)

    # 2. IMAGE MODELS BENCHMARK
    print("\n[Step 2/5] Benchmarking All 5 Image Classification Models...")
    image_candidates = CONFIG.image_models
    image_results = []
    trained_image_models = {}

    for img_model_name in image_candidates:
        trainer = ImageModelTrainer(
            model_name=img_model_name, 
            epochs=epochs, 
            use_focal=CONFIG.use_focal_loss
        )
        metrics, trained_model = trainer.train_and_evaluate(
            train_img_loader, 
            val_img_loader, 
            class_weights=class_weights
        )
        image_results.append(metrics)
        trained_image_models[img_model_name] = trained_model

    # Rank & Compare Top 3 Image Models
    all_img_ranked, top_3_img, best_img_model = comparator.rank_models(image_results)
    img_report = comparator.generate_comparison_report("Image Classification", all_img_ranked, top_3_img, best_img_model)
    comparator.plot_top3_comparison("Image Classification", top_3_img, CONFIG.reports_dir / "top3_image_models_comparison.png")
    print(img_report)

    # 3. TABULAR MODELS BENCHMARK
    print("\n[Step 3/5] Benchmarking All 5 Tabular Classification Models...")
    tabular_candidates = CONFIG.tabular_models
    tabular_results = []
    trained_tab_models = {}
    input_dim = len(feature_cols)

    for tab_model_name in tabular_candidates:
        trainer = TabularModelTrainer(
            model_name=tab_model_name,
            input_dim=input_dim,
            epochs=epochs,
            use_focal=CONFIG.use_focal_loss
        )
        metrics, trained_model = trainer.train_and_evaluate(
            train_tab_loader,
            val_tab_loader,
            class_weights=class_weights
        )
        tabular_results.append(metrics)
        trained_tab_models[tab_model_name] = trained_model

    # Rank & Compare Top 3 Tabular Models
    all_tab_ranked, top_3_tab, best_tab_model = comparator.rank_models(tabular_results)
    tab_report = comparator.generate_comparison_report("Tabular Classification", all_tab_ranked, top_3_tab, best_tab_model)
    comparator.plot_top3_comparison("Tabular Classification", top_3_tab, CONFIG.reports_dir / "top3_tabular_models_comparison.png")
    print(tab_report)

    # 4. MULTIMODAL FUSION (Champion Image Model + Champion Tabular Model)
    print("\n[Step 4/5] Training Multimodal Fusion of Champion Image & Tabular Models...")
    champ_img_name = best_img_model['model_name']
    champ_tab_name = best_tab_model['model_name']
    champ_img_net = trained_image_models[champ_img_name]
    champ_tab_net = trained_tab_models[champ_tab_name]

    fusion_model = MultimodalLSSFusion(
        image_model=champ_img_net,
        tabular_model=champ_tab_net,
        num_classes=CONFIG.num_classes
    ).to(CONFIG.device)

    # Train Fusion Classifier Head
    fusion_optimizer = torch.optim.AdamW(fusion_model.fusion_fc.parameters(), lr=1e-3)
    fusion_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(CONFIG.device))

    fusion_model.train()
    # Freeze encoders for rapid head adaptation
    for p in champ_img_net.parameters():
        p.requires_grad = False
    for p in champ_tab_net.parameters():
        p.requires_grad = False

    for _ in range(3):
        for (img_b, _, _), (tab_b, y_b) in zip(train_img_loader, train_tab_loader):
            b_size = min(img_b.size(0), tab_b.size(0))
            img_b = img_b[:b_size].to(CONFIG.device)
            tab_b = tab_b[:b_size].to(CONFIG.device)
            y_b = y_b[:b_size].to(CONFIG.device)

            fusion_optimizer.zero_grad()
            out = fusion_model(img_b, tab_b)
            loss = fusion_criterion(out, y_b)
            loss.backward()
            fusion_optimizer.step()

    # Evaluate Multimodal on Test Split (unseen patients)
    fusion_model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for (img_b, _, _), (tab_b, y_b) in zip(test_img_loader, test_tab_loader):
            b_size = min(img_b.size(0), tab_b.size(0))
            img_b = img_b[:b_size].to(CONFIG.device)
            tab_b = tab_b[:b_size].to(CONFIG.device)
            out = fusion_model(img_b, tab_b)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(y_b[:b_size].numpy())

    fusion_metrics = compute_fair_metrics(np.array(test_targets), np.array(test_preds), model_name="Multimodal_Fusion")
    print(f"\n--- Multimodal Fusion Test Results (Unseen Patients) ---")
    print(f"  Macro-F1: {fusion_metrics['macro_f1']:.4f}")
    print(f"  Severe Sensitivity: {fusion_metrics['severe_recall']:.4f}")
    print(f"  Accuracy: {fusion_metrics['accuracy']:.4f}")

    torch.save(fusion_model.state_dict(), CONFIG.checkpoints_dir / "multimodal_fusion_champion.pt")

    # 5. GRAD-CAM VISUAL PROOF GENERATION
    print("\n[Step 5/5] Generating Grad-CAM Heatmaps for Visual Anatomical Verification...")
    cam_samples_generated = []
    
    # Select convolutional model with rich spatial feature maps for high-resolution Grad-CAM
    cam_model = champ_img_net
    if champ_img_name == 'vision_transformer' or not hasattr(cam_model, 'get_cam_target_layer') or cam_model.get_cam_target_layer() is None:
        for fallback_name in ['densenet_121', 'efficientnet_b3', 'yolo_classifier', 'unet_classifier']:
            if fallback_name in trained_image_models:
                cam_model = trained_image_models[fallback_name]
                print(f"  Using high-resolution convolutional backbone ({fallback_name}) for canal Grad-CAM.")
                break

    # Ensure gradients are enabled for Grad-CAM
    for p in cam_model.parameters():
        p.requires_grad = True

    for idx in range(min(5, len(test_img_ds))):
        img_tensor, true_label, meta = test_img_ds[idx]
        img_input = img_tensor.unsqueeze(0).to(CONFIG.device)
        
        with torch.no_grad():
            pred_logits = cam_model(img_input)
            pred_idx = torch.argmax(pred_logits, dim=1).item()

        raw_slice = img_tensor[0].cpu().numpy()
        raw_slice = (raw_slice - raw_slice.min()) / (raw_slice.max() - raw_slice.min() + 1e-6)

        save_cam_path = CONFIG.heatmaps_dir / f"gradcam_patient_{meta['study_id']}_{meta['level'].replace('/', '_')}.png"
        generate_and_save_visual_proof(
            model=cam_model,
            img_tensor=img_input,
            raw_slice=raw_slice,
            level=meta['level'],
            true_label=CONFIG.idx_to_class[true_label],
            pred_label=CONFIG.idx_to_class[pred_idx],
            save_path=save_cam_path
        )
        cam_samples_generated.append(str(save_cam_path))
        print(f"  Saved Grad-CAM verification map: {save_cam_path.name}")

    # Compile Final Benchmark Summary
    summary = {
        'image_models': {
            'all_ranked': all_img_ranked,
            'top_3': top_3_img,
            'champion': best_img_model
        },
        'tabular_models': {
            'all_ranked': all_tab_ranked,
            'top_3': top_3_tab,
            'champion': best_tab_model
        },
        'multimodal_fusion': fusion_metrics,
        'gradcam_heatmaps': cam_samples_generated
    }

    summary_path = CONFIG.reports_dir / "benchmark_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Filter non-serializable elements
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll benchmark summaries, confusion matrices, and Grad-CAM maps successfully saved to: {CONFIG.output_dir}")
    return summary

if __name__ == "__main__":
    run_lss_benchmark()
