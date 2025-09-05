#!/usr/bin/env python3
"""
Hypothesis 5: Mean confidence per model and object predicts count accuracy.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from .base_analyzer import BaseHypothesisAnalyzer

class Hypothesis5Analyzer(BaseHypothesisAnalyzer):
    """
    Analyzes the relationship between detection confidence and count accuracy.
    """
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        super().__init__(metrics_db_path, analytics_db_path, output_dir)
        self.hypothesis_name = "H5_pixel_confidence_relationship"
        self.hypothesis_text = "Mean confidence per model and object predicts count accuracy"

    def _cohens_d(self, x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    def run_analysis(self):
        """
        HYPOTHESIS 5: Mean confidence per model and object predicts count accuracy
        
        Simple analysis investigating how mean detection confidence affects
        count accuracy across different models and object types.
        """
        # Filter data with valid bbox information
        valid_data = self.df[self.df['bbox_pixel_ratios'].notna()].copy()
        
        if len(valid_data) == 0:
            return {'verified': False, 'reason': 'No data available'}
        
        # Extract confidence data from bbox information
        confidence_analysis = []
        
        for _, row in valid_data.iterrows():
            try:
                bboxes = json.loads(row['bbox_pixel_ratios'])
                if bboxes:
                    confidences = [bbox.get('confidence', 0) for bbox in bboxes]
                    mean_confidence = np.mean(confidences) if confidences else 0
                    confidence_analysis.append({
                        'model_short': row['model_short'],
                        'target_object': row['target_object'],
                        'count_accuracy': row['count_accuracy'],
                        'mean_confidence': mean_confidence
                    })
            except (json.JSONDecodeError, TypeError):
                continue
        
        if len(confidence_analysis) == 0:
            return {'verified': False, 'reason': 'No confidence data extracted'}
        
        conf_df = pd.DataFrame(confidence_analysis)
        
        # Create 2x2 visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Effect Size Analysis - Mean confidence comparison with error bars
        correct_conf = conf_df[conf_df['count_accuracy'] == 1.0]['mean_confidence']
        incorrect_conf = conf_df[conf_df['count_accuracy'] == 0.0]['mean_confidence']
        
        if len(correct_conf) > 0 and len(incorrect_conf) > 0:
            effect_size = self._cohens_d(correct_conf, incorrect_conf)
            t_stat, p_val = ttest_ind(correct_conf, incorrect_conf, equal_var=False, nan_policy='omit')
            
            ax1.bar(['Correct Count', 'Incorrect Count'], 
                    [correct_conf.mean(), incorrect_conf.mean()],
                    yerr=[correct_conf.std(), incorrect_conf.std()],
                    capsize=5, color=['green', 'red'], alpha=0.7)
            
            ax1.text(0.5, 0.9, f"Cohen's d = {effect_size:.3f}\nt-stat = {t_stat:.2f}, p = {p_val:.3f}",
                     transform=ax1.transAxes, ha='center', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
        else:
            ax1.text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center', transform=ax1.transAxes)
        
        ax1.set_ylabel('Mean Detection Confidence')
        ax1.set_title('Effect Size: Mean Confidence by Count Accuracy\n(Higher confidence predicts correct counts)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance by model
        model_stats = []
        for model in sorted(conf_df['model_short'].unique()):
            model_df = conf_df[conf_df['model_short'] == model]
            model_stats.append({
                'model': model,
                'mean_accuracy': model_df['count_accuracy'].mean(),
                'mean_confidence': model_df['mean_confidence'].mean()
            })
        
        if model_stats:
            model_stats_df = pd.DataFrame(model_stats)
            ax2.bar(model_stats_df['model'], model_stats_df['mean_accuracy'], color='skyblue')
        
        ax2.set_ylabel('Mean Count Accuracy')
        ax2.set_title('Performance by Model', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance by object type
        object_stats = []
        for obj in sorted(conf_df['target_object'].unique()):
            obj_df = conf_df[conf_df['target_object'] == obj]
            object_stats.append({
                'object': obj,
                'mean_accuracy': obj_df['count_accuracy'].mean(),
                'mean_confidence': obj_df['mean_confidence'].mean()
            })
        
        if object_stats:
            object_stats_df = pd.DataFrame(object_stats)
            ax3.bar(object_stats_df['object'], object_stats_df['mean_accuracy'], color='lightgreen')
        
        ax3.set_ylabel('Mean Count Accuracy')
        ax3.set_title('Performance by Object Type', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model-Object heatmap
        model_object_pivot = conf_df.groupby(['model_short', 'target_object']).agg({
            'count_accuracy': 'mean',
            'mean_confidence': 'mean'
        }).round(3)
        
        if len(model_object_pivot) > 0:
            accuracy_pivot = model_object_pivot['count_accuracy'].unstack()
            sns.heatmap(accuracy_pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax4, linewidths=.5)
        
        ax4.set_title('Model-Object Performance Heatmap', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_5_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical analysis - use effect size instead of correlation for binary outcome
        confidence_significant = False
        effect_size = 0
        t_stat = 0
        p_val = 1.0
        
        if len(correct_conf) > 0 and len(incorrect_conf) > 0:
            t_stat, p_val = ttest_ind(correct_conf, incorrect_conf, equal_var=False, nan_policy='omit')
            effect_size = self._cohens_d(correct_conf, incorrect_conf)
            confidence_significant = p_val < 0.05 and abs(effect_size) > 0.2
        
        result = {
            'verified': confidence_significant,
            'effect_size': {'cohens_d': effect_size, 't_stat': t_stat, 'p_val': p_val} if len(correct_conf) > 0 and len(incorrect_conf) > 0 else None,
            'model_stats': model_stats,
            'object_stats': object_stats,
            'sample_size': len(conf_df),
            'correct_samples': len(correct_conf) if len(correct_conf) > 0 else 0,
            'incorrect_samples': len(incorrect_conf) if len(incorrect_conf) > 0 else 0,
            'conclusion': f"{'✅ VERIFIED' if confidence_significant else '❌ NOT VERIFIED'}: "
                         f"Effect size (Cohen's d): {effect_size:.3f}, p={p_val:.3f}" if len(correct_conf) > 0 and len(incorrect_conf) > 0 else "Insufficient data"
        }
        
        self.hypothesis_results[self.hypothesis_name] = result
        return result
