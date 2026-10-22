"""
Model Comparison & Selection Engine for Lumbar Spinal Stenosis.
Ranks all 5 Image and all 5 Tabular models, selects Top 3 for each modality,
conducts head-to-head comparative analysis, and elects the overall best model.
"""
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from config.config import CONFIG

class ModelComparator:
    """
    Ranks models according to clinically safe metrics:
    Macro-F1 (primary) + Severe Recall (critical) + Low Extreme Error Rate.
    """
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or CONFIG.reports_dir

    def calculate_clinical_score(self, metrics: Dict) -> float:
        """
        Calculates a unified clinical score:
        - 40% Macro-F1 (balanced detection across all 3 grades)
        - 35% Severe Sensitivity / Recall (clinical safety: never miss severe stenosis)
        - 15% Overall Accuracy
        - 10% Penalty reduction for extreme errors (Normal confused with Severe)
        """
        f1 = metrics.get('macro_f1', 0.0)
        sev_r = metrics.get('severe_recall', 0.0)
        acc = metrics.get('accuracy', 0.0)
        extreme_err = metrics.get('extreme_error_rate', 0.0)

        score = (0.40 * f1) + (0.35 * sev_r) + (0.15 * acc) - (0.10 * extreme_err)
        return float(score)

    def rank_models(self, benchmark_results: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Ranks all candidate models, extracts Top 3, and selects the Best model.
        Returns:
            all_ranked: List of all models sorted by clinical score
            top_3: List of top 3 models
            best_model: The single champion model
        """
        for res in benchmark_results:
            res['clinical_score'] = self.calculate_clinical_score(res)

        # Sort descending by clinical score, then macro_f1
        all_ranked = sorted(
            benchmark_results, 
            key=lambda m: (m['clinical_score'], m.get('macro_f1', 0.0), m.get('severe_recall', 0.0)), 
            reverse=True
        )

        for rank, m in enumerate(all_ranked, 1):
            m['rank'] = rank

        top_3 = all_ranked[:3]
        best_model = top_3[0]

        return all_ranked, top_3, best_model

    def generate_comparison_report(
        self, 
        modality: str, 
        all_ranked: List[Dict], 
        top_3: List[Dict], 
        best_model: Dict
    ) -> str:
        """
        Formats a comprehensive markdown and ASCII comparison report.
        """
        headers = ["Rank", "Model", "Macro-F1", "Severe Recall", "Accuracy", "Extreme Err", "Clinical Score"]
        
        table_data = []
        for m in all_ranked:
            table_data.append([
                f"#{m['rank']}",
                m['model_name'],
                f"{m.get('macro_f1', 0.0):.4f}",
                f"{m.get('severe_recall', 0.0):.4f}",
                f"{m.get('accuracy', 0.0):.4f}",
                f"{m.get('extreme_error_rate', 0.0):.4f}",
                f"{m.get('clinical_score', 0.0):.4f}"
            ])

        ascii_table = tabulate(table_data, headers=headers, tablefmt="github")

        # Top 3 Head-to-Head Deep Dive
        top3_headers = ["Metric / Attribute", top_3[0]['model_name'], top_3[1]['model_name'], top_3[2]['model_name']]
        top3_rows = [
            ["Macro-F1 Score", f"{top_3[0].get('macro_f1', 0):.4f}", f"{top_3[1].get('macro_f1', 0):.4f}", f"{top_3[2].get('macro_f1', 0):.4f}"],
            ["Severe Stenosis Sensitivity", f"{top_3[0].get('severe_recall', 0):.4f}", f"{top_3[1].get('severe_recall', 0):.4f}", f"{top_3[2].get('severe_recall', 0):.4f}"],
            ["Moderate Stenosis Sensitivity", f"{top_3[0].get('moderate_recall', 0):.4f}", f"{top_3[1].get('moderate_recall', 0):.4f}", f"{top_3[2].get('moderate_recall', 0):.4f}"],
            ["Normal/Mild Sensitivity", f"{top_3[0].get('normal_recall', 0):.4f}", f"{top_3[1].get('normal_recall', 0):.4f}", f"{top_3[2].get('normal_recall', 0):.4f}"],
            ["Adjacent-Grade Confusion", f"{top_3[0].get('adjacent_error_rate', 0):.4f}", f"{top_3[1].get('adjacent_error_rate', 0):.4f}", f"{top_3[2].get('adjacent_error_rate', 0):.4f}"],
            ["Overall Clinical Score", f"{top_3[0].get('clinical_score', 0):.4f}", f"{top_3[1].get('clinical_score', 0):.4f}", f"{top_3[2].get('clinical_score', 0):.4f}"],
        ]
        top3_ascii = tabulate(top3_rows, headers=top3_headers, tablefmt="github")

        report = f"""
================================================================================
   {modality.upper()} BENCHMARK & COMPARISON REPORT: LUMBAR SPINAL STENOSIS
================================================================================

--- COMPLETE CANDIDATE RANKINGS (All 5 Models) ---
{ascii_table}

--- TOP 3 HEAD-TO-HEAD COMPARISON ---
{top3_ascii}

================================================================================
>>> CHAMPION WINNER: {best_model['model_name']}
  * Macro-F1: {best_model.get('macro_f1', 0.0):.4f}
  * Severe Recall (Clinical Safety): {best_model.get('severe_recall', 0.0):.4f}
  * Clinical Selection Rationale: Highest composite score balancing detection
    of rare dangerous severe stenosis cases with minimal adjacent-grade misclassification.
================================================================================
"""
        # Save text report to disk
        report_file = self.output_dir / f"{modality.lower()}_comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        return report

    def plot_top3_comparison(self, modality: str, top_3: List[Dict], save_path: Path):
        """Generates a comparison bar chart across the Top 3 models."""
        metrics_to_plot = ['macro_f1', 'severe_recall', 'accuracy', 'clinical_score']
        labels = ['Macro-F1', 'Severe Recall', 'Accuracy', 'Clinical Score']

        model_names = [m['model_name'] for m in top_3]
        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, m in enumerate(top_3):
            vals = [m.get(k, 0.0) for k in metrics_to_plot]
            ax.bar(x + (i - 1) * width, vals, width, label=f"#{i+1}: {m['model_name']}", color=colors[i], alpha=0.9)

        ax.set_ylabel('Score / Proportion', fontsize=11)
        ax.set_title(f'Top 3 {modality.title()} Models Head-to-Head Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=200)
        plt.close()
