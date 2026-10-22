"""
Web Application for Lumbar Spinal Stenosis (LSS) Diagnostic & Benchmark Platform.
"""
from typing import Dict, List, Optional
import os
import shutil
from pathlib import Path
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from config.config import CONFIG
from inference.predict import LSSClinicalPredictor
from evaluation.gradcam import generate_and_save_visual_proof

app = Flask(
    __name__, 
    template_folder=str(CONFIG.project_root / "templates"),
    static_folder=str(CONFIG.project_root / "static")
)
app.config['UPLOAD_FOLDER'] = str(CONFIG.output_dir / "uploads")
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Cache predictor instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        # Check if champion model checkpoint exists, default to densenet_121 or efficientnet_b3
        champ_name = "densenet_121"
        summary_path = CONFIG.reports_dir / "benchmark_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    champ_name = data.get('image_models', {}).get('champion', {}).get('model_name', 'densenet_121')
            except Exception:
                pass
        predictor = LSSClinicalPredictor(model_name=champ_name)
    return predictor

def load_benchmark_summary():
    summary_path = CONFIG.reports_dir / "benchmark_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {
        'image_models': {'all_ranked': [], 'top_3': [], 'champion': {}},
        'tabular_models': {'all_ranked': [], 'top_3': [], 'champion': {}},
        'multimodal_fusion': {},
        'gradcam_heatmaps': []
    }

@app.route('/')
def index():
    summary = load_benchmark_summary()
    img_ranked = summary.get('image_models', {}).get('all_ranked', [])
    tab_ranked = summary.get('tabular_models', {}).get('all_ranked', [])

    top3_img_plot = CONFIG.reports_dir / "top3_image_models_comparison.png"
    top3_tab_plot = CONFIG.reports_dir / "top3_tabular_models_comparison.png"

    top3_img_url = f"/reports/{top3_img_plot.name}" if top3_img_plot.exists() else None
    top3_tab_url = f"/reports/{top3_tab_plot.name}" if top3_tab_plot.exists() else None

    return render_template(
        'index.html',
        image_models_ranked=img_ranked,
        tabular_models_ranked=tab_ranked,
        top3_img_plot_url=top3_img_url,
        top3_tab_plot_url=top3_tab_url,
        result=None
    )

@app.route('/predict', methods=['POST'])
def handle_predict():
    if 'mri_file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['mri_file']
    if file.filename == '':
        return redirect(url_for('index'))

    level = request.form.get('level', 'L4/L5')
    filename = secure_filename(file.filename)
    save_path = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(str(save_path))

    pred_engine = get_predictor()
    finding = pred_engine.predict_slice(save_path, level=level, generate_cam=True)

    # Save visual views (Full Spine MRI, Canal Zoom, Grad-CAM Overlay, 3-Panel Composite)
    stem = save_path.stem
    if 'composite_bgr' in finding:
        comp_filename = f"composite_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / comp_filename), finding['composite_bgr'])
        finding['composite_url'] = f"/heatmaps/{comp_filename}"

    if 'full_annotated_bgr' in finding:
        full_filename = f"full_spine_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / full_filename), finding['full_annotated_bgr'])
        finding['full_url'] = f"/heatmaps/{full_filename}"

    if 'canal_zoom_bgr' in finding:
        zoom_filename = f"canal_zoom_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / zoom_filename), finding['canal_zoom_bgr'])
        finding['canal_zoom_url'] = f"/heatmaps/{zoom_filename}"

    if 'overlay_bgr' in finding:
        cam_filename = f"cam_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / cam_filename), finding['overlay_bgr'])
        finding['cam_url'] = f"/heatmaps/{cam_filename}"

    summary = load_benchmark_summary()
    return render_template(
        'index.html',
        image_models_ranked=summary.get('image_models', {}).get('all_ranked', []),
        tabular_models_ranked=summary.get('tabular_models', {}).get('all_ranked', []),
        top3_img_plot_url=f"/reports/top3_image_models_comparison.png",
        top3_tab_plot_url=f"/reports/top3_tabular_models_comparison.png",
        result=finding
    )

@app.route('/predict_sample')
def predict_sample():
    # Find any sample DICOM in dataset
    sample_files = list(CONFIG.raw_dicom_dir.glob("**/*.dcm"))
    if not sample_files:
        return redirect(url_for('index'))

    sample_dcm = sample_files[len(sample_files) // 2]
    pred_engine = get_predictor()
    finding = pred_engine.predict_slice(sample_dcm, level="L4/L5", generate_cam=True)

    # Save visual views for sample
    stem = sample_dcm.stem
    if 'composite_bgr' in finding:
        comp_filename = f"composite_sample_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / comp_filename), finding['composite_bgr'])
        finding['composite_url'] = f"/heatmaps/{comp_filename}"

    if 'full_annotated_bgr' in finding:
        full_filename = f"full_spine_sample_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / full_filename), finding['full_annotated_bgr'])
        finding['full_url'] = f"/heatmaps/{full_filename}"

    if 'canal_zoom_bgr' in finding:
        zoom_filename = f"canal_zoom_sample_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / zoom_filename), finding['canal_zoom_bgr'])
        finding['canal_zoom_url'] = f"/heatmaps/{zoom_filename}"

    if 'overlay_bgr' in finding:
        cam_filename = f"cam_sample_{stem}.png"
        cv2.imwrite(str(CONFIG.heatmaps_dir / cam_filename), finding['overlay_bgr'])
        finding['cam_url'] = f"/heatmaps/{cam_filename}"

    summary = load_benchmark_summary()
    return render_template(
        'index.html',
        image_models_ranked=summary.get('image_models', {}).get('all_ranked', []),
        tabular_models_ranked=summary.get('tabular_models', {}).get('all_ranked', []),
        top3_img_plot_url=f"/reports/top3_image_models_comparison.png",
        top3_tab_plot_url=f"/reports/top3_tabular_models_comparison.png",
        result=finding
    )

@app.route('/heatmaps/<filename>')
def serve_heatmap(filename):
    return send_from_directory(str(CONFIG.heatmaps_dir), filename)

@app.route('/reports/<filename>')
def serve_report_file(filename):
    return send_from_directory(str(CONFIG.reports_dir), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
