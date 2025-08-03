#!/usr/bin/env python3
"""
Advanced Count Adherence Analysis for Diffusion Models

This script provides comprehensive analytical insights into diffusion model performance
for count adherence tasks. It analyzes 4800 generated images from 4 diffusion models
across multiple target objects, variants, and expected counts.

Key analyses:
1. Average accuracy per number of desired objects per target class per model
2. Segmentation pixel ratio analysis for accurate vs inaccurate generations  
3. Advanced statistical insights and correlations
4. Performance degradation analysis
5. Variant comparison analysis
6. Object-specific difficulty assessment
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats

from matplotlib.ticker import MaxNLocator

# Configure plotting
plt.style.use('default')
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (12, 8)
})

# Constants
IGNORED_OBJECTS = {"brocolli"}  # Known to be problematic in original analysis
MODEL_COLORS = {
    'stable-diffusion-v1-5': '#1f77b4',
    'stable-diffusion-2-1': '#ff7f0e', 
    'stable-diffusion-3-medium': '#2ca02c',
    'stable-diffusion-3.5-medium': '#d62728'
}

class CountAnalyzer:
    """Advanced analyzer for count adherence in diffusion models."""
    
    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.df = None
        self.filtered_df = None
        
    def load_data(self):
        """Load and preprocess data from SQLite database."""
        print("Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        
        # Load comprehensive data including new bbox metrics
        self.df = pd.read_sql_query("""
            SELECT 
                model_name, variant, target_object, expected_count, detected_count,
                count_accuracy, pixel_ratio, coverage_percentage, 
                target_pixels, total_pixels, image_path,
                guidance_scale, num_inference_steps, condition_type,
                segmented_pixels_in_bbox, total_bbox_pixels,
                bbox_pixel_ratios, min_bbox_pixel_ratio, max_bbox_pixel_ratio, std_bbox_pixel_ratio
            FROM image_evaluations
            WHERE expected_count IS NOT NULL AND count_accuracy IS NOT NULL
        """, conn)
        conn.close()
        
        # Clean model names for better visualization
        self.df['model_short'] = (
            self.df['model_name']
            .str.split('/').str[-1]
            .str.replace('stable-diffusion-', 'SD-')
            .str.replace('-diffusers', '')
        )
        
        # Filter out problematic objects and invalid data
        self.filtered_df = self.df[
            (self.df['count_accuracy'] >= 0) &
            (self.df['count_accuracy'] <= 1) &
            (~self.df['target_object'].isin(IGNORED_OBJECTS))
        ].copy()
        
        print(f"Loaded {len(self.df)} total evaluations")
        print(f"After filtering: {len(self.filtered_df)} evaluations")
        print(f"Models: {self.filtered_df['model_short'].unique()}")
        print(f"Objects: {self.filtered_df['target_object'].unique()}")
        print(f"Variants: {self.filtered_df['variant'].unique()}")
        
    def analyze_accuracy_by_count_and_object(self):
        """Analysis 1: Average accuracy per number of desired objects per target class per model."""
        print("\n=== Analysis 1: Accuracy by Count, Object, and Model ===")
        
        # Group and calculate mean accuracy
        grouped = (
            self.filtered_df.groupby(['model_short', 'target_object', 'expected_count'])
            .agg({
                'count_accuracy': ['mean', 'std', 'count'],
                'detected_count': 'mean'
            })
            .round(4)
        )
        grouped.columns = ['accuracy_mean', 'accuracy_std', 'n_samples', 'avg_detected']
        grouped = grouped.reset_index()
        
        # Save detailed numerical results
        results_file = self.output_dir / "accuracy_by_count_object_model.csv"
        grouped.to_csv(results_file, index=False)
        print(f"Detailed results saved to: {results_file}")
        
        # Create individual plots for each model
        models = grouped['model_short'].unique()
        for model in models:
            model_data = grouped[grouped['model_short'] == model]
            
            plt.figure(figsize=(14, 10))
            
            # Plot accuracy lines for each object
            objects = model_data['target_object'].unique()
            for i, obj in enumerate(objects):
                obj_data = model_data[model_data['target_object'] == obj]
                plt.plot(obj_data['expected_count'], obj_data['accuracy_mean'] * 100, 
                        marker='o', linewidth=2, markersize=6, label=obj.capitalize())
                
                # Add error bars if we have std data
                plt.errorbar(obj_data['expected_count'], obj_data['accuracy_mean'] * 100,
                           yerr=obj_data['accuracy_std'] * 100, alpha=0.3, capsize=3)
            
            plt.xlabel("Expected Number of Objects")
            plt.ylabel("Count Accuracy (%)")
            plt.title(f"Count Accuracy vs Expected Count - {model}")
            plt.legend(title="Target Object", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            
            plot_path = self.output_dir / f"accuracy_detailed_{model.replace('/', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_path}")
        
        # Create summary comparison plot
        self._create_model_comparison_plot(grouped)
        
        # Statistical analysis
        self._statistical_analysis_accuracy(grouped)
        
    def _create_model_comparison_plot(self, grouped_data):
        """Create model comparison plot averaged across objects."""
        model_avg = (
            grouped_data.groupby(['model_short', 'expected_count'])
            .agg({'accuracy_mean': 'mean', 'n_samples': 'sum'})
            .reset_index()
        )
        
        plt.figure(figsize=(12, 8))
        for model in model_avg['model_short'].unique():
            model_data = model_avg[model_avg['model_short'] == model]
            plt.plot(model_data['expected_count'], model_data['accuracy_mean'] * 100, 
                    marker='o', linewidth=3, markersize=8, label=model)
        
        plt.xlabel("Expected Number of Objects")
        plt.ylabel("Average Count Accuracy (%)")
        plt.title("Model Comparison: Average Count Accuracy Across All Objects")
        plt.legend(title="Model")
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        
        plot_path = self.output_dir / "model_comparison_average.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {plot_path}")
        
    def _statistical_analysis_accuracy(self, grouped_data):
        """Perform statistical analysis on accuracy data."""
        print("\n--- Statistical Analysis ---")
        
        # Find best and worst performing combinations
        best_performance = grouped_data.loc[grouped_data['accuracy_mean'].idxmax()]
        worst_performance = grouped_data.loc[grouped_data['accuracy_mean'].idxmin()]
        
        print(f"Best performance: {best_performance['model_short']} - {best_performance['target_object']} - "
              f"Count {best_performance['expected_count']} - Accuracy: {best_performance['accuracy_mean']:.3f}")
        print(f"Worst performance: {worst_performance['model_short']} - {worst_performance['target_object']} - "
              f"Count {worst_performance['expected_count']} - Accuracy: {worst_performance['accuracy_mean']:.3f}")
        
        # Analyze performance degradation with count
        degradation_analysis = []
        for model in grouped_data['model_short'].unique():
            for obj in grouped_data['target_object'].unique():
                subset = grouped_data[
                    (grouped_data['model_short'] == model) & 
                    (grouped_data['target_object'] == obj)
                ]
                if len(subset) > 1:
                    # Calculate correlation between expected_count and accuracy
                    corr, p_value = stats.pearsonr(subset['expected_count'], subset['accuracy_mean'])
                    degradation_analysis.append({
                        'model': model,
                        'object': obj,
                        'correlation': corr,
                        'p_value': p_value,
                        'degradation': corr < -0.5  # Strong negative correlation indicates degradation
                    })
        
        degradation_df = pd.DataFrame(degradation_analysis)
        degradation_file = self.output_dir / "performance_degradation_analysis.csv"
        degradation_df.to_csv(degradation_file, index=False)
        print(f"Performance degradation analysis saved to: {degradation_file}")
        
    def analyze_pixel_ratio_accuracy_correlation(self):
        """Analysis 2: Compare segmentation pixel ratio for accurate vs inaccurate generations."""
        print("\n=== Analysis 2: Pixel Ratio vs Accuracy Analysis ===")
        
        # Filter data with valid pixel information
        pixel_data = self.filtered_df[
            (self.filtered_df['pixel_ratio'].notna()) &
            (self.filtered_df['coverage_percentage'].notna())
        ].copy()
        
        # Create accuracy categories
        pixel_data['accuracy_category'] = pixel_data['count_accuracy'].map({1.0: 'Accurate', 0.0: 'Inaccurate'})
        pixel_data = pixel_data[pixel_data['accuracy_category'].notna()]
        
        print(f"Analyzing {len(pixel_data)} images with pixel data")
        print(f"Accurate: {len(pixel_data[pixel_data['accuracy_category'] == 'Accurate'])}")
        print(f"Inaccurate: {len(pixel_data[pixel_data['accuracy_category'] == 'Inaccurate'])}")
        
        # Statistical comparison
        accurate_pixels = pixel_data[pixel_data['accuracy_category'] == 'Accurate']['pixel_ratio']
        inaccurate_pixels = pixel_data[pixel_data['accuracy_category'] == 'Inaccurate']['pixel_ratio']
        
        # Perform statistical tests
        t_stat, t_p = stats.ttest_ind(accurate_pixels, inaccurate_pixels)
        u_stat, u_p = stats.mannwhitneyu(accurate_pixels, inaccurate_pixels, alternative='two-sided')
        
        print("\n--- Statistical Test Results ---")
        print(f"Accurate generations - Pixel ratio: {accurate_pixels.mean():.4f} ± {accurate_pixels.std():.4f}")
        print(f"Inaccurate generations - Pixel ratio: {inaccurate_pixels.mean():.4f} ± {inaccurate_pixels.std():.4f}")
        print(f"T-test: t={t_stat:.4f}, p={t_p:.4e}")
        print(f"Mann-Whitney U test: U={u_stat:.4f}, p={u_p:.4e}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot comparison
        pixel_data.boxplot(column='pixel_ratio', by='accuracy_category', ax=axes[0,0])
        axes[0,0].set_title('Pixel Ratio Distribution by Accuracy')
        axes[0,0].set_xlabel('Accuracy Category')
        axes[0,0].set_ylabel('Pixel Ratio')
        
        # Distribution plots
        accurate_pixels.hist(bins=30, alpha=0.7, label='Accurate', ax=axes[0,1])
        inaccurate_pixels.hist(bins=30, alpha=0.7, label='Inaccurate', ax=axes[0,1])
        axes[0,1].set_title('Pixel Ratio Distributions')
        axes[0,1].set_xlabel('Pixel Ratio')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Coverage percentage comparison
        pixel_data.boxplot(column='coverage_percentage', by='accuracy_category', ax=axes[1,0])
        axes[1,0].set_title('Coverage Percentage by Accuracy')
        axes[1,0].set_xlabel('Accuracy Category')
        axes[1,0].set_ylabel('Coverage Percentage (%)')
        
        # Scatter plot: pixel ratio vs expected count, colored by accuracy
        for accuracy in ['Accurate', 'Inaccurate']:
            subset = pixel_data[pixel_data['accuracy_category'] == accuracy]
            axes[1,1].scatter(subset['expected_count'], subset['pixel_ratio'], 
                            alpha=0.6, label=accuracy, s=30)
        axes[1,1].set_title('Pixel Ratio vs Expected Count')
        axes[1,1].set_xlabel('Expected Count')
        axes[1,1].set_ylabel('Pixel Ratio')
        axes[1,1].legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / "pixel_ratio_accuracy_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved pixel analysis plot: {plot_path}")
        
        # Detailed breakdown by model and object
        self._detailed_pixel_analysis(pixel_data)
        
    def _detailed_pixel_analysis(self, pixel_data):
        """Detailed pixel analysis breakdown."""
        # Group by model and object
        detailed_analysis = []
        
        for model in pixel_data['model_short'].unique():
            for obj in pixel_data['target_object'].unique():
                subset = pixel_data[
                    (pixel_data['model_short'] == model) & 
                    (pixel_data['target_object'] == obj)
                ]
                
                if len(subset) < 10:  # Skip if insufficient data
                    continue
                
                accurate = subset[subset['accuracy_category'] == 'Accurate']
                inaccurate = subset[subset['accuracy_category'] == 'Inaccurate']
                
                if len(accurate) > 0 and len(inaccurate) > 0:
                    t_stat, p_val = stats.ttest_ind(accurate['pixel_ratio'], inaccurate['pixel_ratio'])
                    
                    detailed_analysis.append({
                        'model': model,
                        'object': obj,
                        'n_accurate': len(accurate),
                        'n_inaccurate': len(inaccurate),
                        'accurate_pixel_mean': accurate['pixel_ratio'].mean(),
                        'accurate_pixel_std': accurate['pixel_ratio'].std(),
                        'inaccurate_pixel_mean': inaccurate['pixel_ratio'].mean(),
                        'inaccurate_pixel_std': inaccurate['pixel_ratio'].std(),
                        'pixel_ratio_t_stat': t_stat,
                        'pixel_ratio_p_value': p_val,
                        'significant_difference': p_val < 0.05
                    })
        
        detailed_df = pd.DataFrame(detailed_analysis)
        detailed_file = self.output_dir / "detailed_pixel_ratio_analysis.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Detailed pixel analysis saved to: {detailed_file}")
        
    def advanced_insights_analysis(self):
        """Analysis 3: Advanced statistical insights and patterns."""
        print("\n=== Analysis 3: Advanced Insights ===")
        
        # 1. Variant Performance Analysis
        self._analyze_variant_performance()
        
        # 2. Object Difficulty Ranking
        self._analyze_object_difficulty()
        
        # 3. Count vs Detection Error Analysis
        self._analyze_count_detection_errors()
        
        # 4. Model Consistency Analysis
        self._analyze_model_consistency()
        
        # 5. Correlation Analysis
        self._correlation_analysis()
        
        # 6. Outlier Detection Analysis
        self._analyze_outliers_and_extreme_cases()
        
    def _analyze_variant_performance(self):
        """Analyze performance differences between variants."""
        print("\n--- Variant Performance Analysis ---")
        
        variant_stats = (
            self.filtered_df.groupby(['model_short', 'variant'])
            .agg({
                'count_accuracy': ['mean', 'std', 'count'],
                'pixel_ratio': 'mean',
                'coverage_percentage': 'mean'
            })
            .round(4)
        )
        variant_stats.columns = ['accuracy_mean', 'accuracy_std', 'n_samples', 'pixel_ratio_mean', 'coverage_mean']
        variant_stats = variant_stats.reset_index()
        
        # Create variant comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy by variant
        for model in variant_stats['model_short'].unique():
            model_data = variant_stats[variant_stats['model_short'] == model]
            axes[0].plot(model_data['variant'], model_data['accuracy_mean'], 
                        marker='o', linewidth=2, label=model, markersize=8)
        
        axes[0].set_title('Count Accuracy by Variant')
        axes[0].set_xlabel('Variant')
        axes[0].set_ylabel('Mean Count Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Pixel ratio by variant
        for model in variant_stats['model_short'].unique():
            model_data = variant_stats[variant_stats['model_short'] == model]
            axes[1].plot(model_data['variant'], model_data['pixel_ratio_mean'], 
                        marker='s', linewidth=2, label=model, markersize=8)
        
        axes[1].set_title('Pixel Ratio by Variant')
        axes[1].set_xlabel('Variant')
        axes[1].set_ylabel('Mean Pixel Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "variant_performance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        variant_file = self.output_dir / "variant_performance_stats.csv"
        variant_stats.to_csv(variant_file, index=False)
        print(f"Variant analysis saved to: {variant_file}")
        
    def _analyze_object_difficulty(self):
        """Rank objects by difficulty across models."""
        print("\n--- Object Difficulty Analysis ---")
        
        object_difficulty = (
            self.filtered_df.groupby('target_object')
            .agg({
                'count_accuracy': ['mean', 'std'],
                'pixel_ratio': 'mean',
                'expected_count': 'mean'
            })
            .round(4)
        )
        object_difficulty.columns = ['accuracy_mean', 'accuracy_std', 'pixel_ratio_mean', 'avg_expected_count']
        object_difficulty = object_difficulty.reset_index()
        object_difficulty = object_difficulty.sort_values('accuracy_mean')
        
        print("Object Difficulty Ranking (easiest to hardest):")
        for idx, row in object_difficulty.iterrows():
            print(f"{row['target_object']:10}: {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(object_difficulty['target_object'], object_difficulty['accuracy_mean'], 
                      yerr=object_difficulty['accuracy_std'], capsize=5, alpha=0.8)
        plt.title('Object Difficulty Ranking by Average Accuracy')
        plt.xlabel('Target Object')
        plt.ylabel('Mean Count Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Color bars by difficulty
        colors = plt.cm.RdYlGn(object_difficulty['accuracy_mean'])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plot_path = self.output_dir / "object_difficulty_ranking.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        difficulty_file = self.output_dir / "object_difficulty_analysis.csv"
        object_difficulty.to_csv(difficulty_file, index=False)
        print(f"Object difficulty analysis saved to: {difficulty_file}")
        
    def _analyze_count_detection_errors(self):
        """Analyze patterns in count detection errors."""
        print("\n--- Count Detection Error Analysis ---")
        
        # Calculate detection errors
        error_data = self.filtered_df.copy()
        error_data['detection_error'] = error_data['detected_count'] - error_data['expected_count']
        error_data['abs_error'] = error_data['detection_error'].abs()
        
        # Error patterns by expected count
        error_by_count = (
            error_data.groupby(['expected_count', 'model_short'])
            .agg({
                'detection_error': ['mean', 'std'],
                'abs_error': 'mean',
                'count_accuracy': 'mean'
            })
            .round(3)
        )
        error_by_count.columns = ['mean_error', 'std_error', 'mean_abs_error', 'accuracy']
        error_by_count = error_by_count.reset_index()
        
        # Create error analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mean error by expected count
        for model in error_by_count['model_short'].unique():
            model_data = error_by_count[error_by_count['model_short'] == model]
            axes[0,0].plot(model_data['expected_count'], model_data['mean_error'], 
                          marker='o', label=model, linewidth=2)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0,0].set_title('Mean Detection Error by Expected Count')
        axes[0,0].set_xlabel('Expected Count')
        axes[0,0].set_ylabel('Mean Error (Detected - Expected)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Absolute error
        for model in error_by_count['model_short'].unique():
            model_data = error_by_count[error_by_count['model_short'] == model]
            axes[0,1].plot(model_data['expected_count'], model_data['mean_abs_error'], 
                          marker='s', label=model, linewidth=2)
        axes[0,1].set_title('Mean Absolute Error by Expected Count')
        axes[0,1].set_xlabel('Expected Count')
        axes[0,1].set_ylabel('Mean Absolute Error')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Error distribution histogram
        error_data['detection_error'].hist(bins=range(-10, 11), alpha=0.7, ax=axes[1,0])
        axes[1,0].set_title('Distribution of Detection Errors')
        axes[1,0].set_xlabel('Detection Error (Detected - Expected)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Confusion matrix style: expected vs detected
        confusion_data = error_data.groupby(['expected_count', 'detected_count']).size().unstack(fill_value=0)
        axes[1,1].imshow(confusion_data.values, cmap='Blues', aspect='auto')
        axes[1,1].set_title('Expected vs Detected Count Matrix')
        axes[1,1].set_xlabel('Detected Count')
        axes[1,1].set_ylabel('Expected Count')
        axes[1,1].set_xticks(range(len(confusion_data.columns)))
        axes[1,1].set_xticklabels(confusion_data.columns)
        axes[1,1].set_yticks(range(len(confusion_data.index)))
        axes[1,1].set_yticklabels(confusion_data.index)
        
        # Add text annotations
        for i in range(len(confusion_data.index)):
            for j in range(len(confusion_data.columns)):
                axes[1,1].text(j, i, str(confusion_data.values[i, j]), 
                             ha='center', va='center', color='black')
        
        plt.tight_layout()
        plot_path = self.output_dir / "count_detection_error_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        error_file = self.output_dir / "count_detection_error_stats.csv"
        error_by_count.to_csv(error_file, index=False)
        print(f"Error analysis saved to: {error_file}")
        
    def _analyze_model_consistency(self):
        """Analyze consistency of model performance across conditions."""
        print("\n--- Model Consistency Analysis ---")
        
        # Calculate coefficient of variation for each model across different conditions
        consistency_data = []
        
        for model in self.filtered_df['model_short'].unique():
            model_data = self.filtered_df[self.filtered_df['model_short'] == model]
            
            # Group by different condition combinations
            for grouping in [['target_object'], ['expected_count'], ['variant'], ['target_object', 'expected_count']]:
                grouped = model_data.groupby(grouping)['count_accuracy'].mean()
                
                consistency_data.append({
                    'model': model,
                    'grouping': '_'.join(grouping),
                    'mean_accuracy': grouped.mean(),
                    'std_accuracy': grouped.std(),
                    'cv_accuracy': grouped.std() / grouped.mean() if grouped.mean() > 0 else np.nan,
                    'min_accuracy': grouped.min(),
                    'max_accuracy': grouped.max(),
                    'range_accuracy': grouped.max() - grouped.min()
                })
        
        consistency_df = pd.DataFrame(consistency_data)
        consistency_file = self.output_dir / "model_consistency_analysis.csv"
        consistency_df.to_csv(consistency_file, index=False)
        print(f"Model consistency analysis saved to: {consistency_file}")
        
    def _correlation_analysis(self):
        """Analyze correlations between different metrics."""
        print("\n--- Correlation Analysis ---")
        
        # Select numerical columns for correlation
        correlation_cols = ['expected_count', 'detected_count', 'count_accuracy', 
                           'pixel_ratio', 'coverage_percentage', 'target_pixels']
        
        corr_data = self.filtered_df[correlation_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        # Create masked correlation matrix for lower triangle
        masked_corr = corr_data.mask(mask)
        
        # Create heatmap
        im = plt.imshow(masked_corr.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Key Metrics')
        
        # Set ticks and labels
        plt.xticks(range(len(corr_data.columns)), corr_data.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr_data.index)), corr_data.index)
        
        # Add correlation values as text
        for i in range(len(corr_data.index)):
            for j in range(len(corr_data.columns)):
                if not mask[i, j] and not np.isnan(masked_corr.values[i, j]):
                    plt.text(j, i, f'{corr_data.values[i, j]:.2f}', 
                           ha='center', va='center', color='black', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, shrink=0.8)
        plt.tight_layout()
        
        plot_path = self.output_dir / "correlation_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        corr_file = self.output_dir / "correlation_matrix.csv"
        corr_data.to_csv(corr_file)
        print(f"Correlation analysis saved to: {corr_file}")
        
    def _analyze_outliers_and_extreme_cases(self):
        """Detect and analyze outliers and extreme cases, including per-bbox pixel ratio analysis."""
        print("\n--- Enhanced Outlier Detection with Per-BBox Analysis ---")
        
        # Debug model distribution first
        print(f"Models in dataset: {sorted(self.filtered_df['model_short'].unique())}")
        model_dist = self.filtered_df['model_short'].value_counts()
        print("Model distribution:")
        for model, count in model_dist.items():
            print(f"  {model}: {count} images")
        
        outliers = []
        
        # 1. Extreme count detection errors with per-bbox analysis
        error_data = self.filtered_df.copy()
        error_data['detection_error'] = error_data['detected_count'] - error_data['expected_count']
        error_data['abs_error'] = error_data['detection_error'].abs()
        
        # Focus on overdetection cases (+1, +2) for bbox analysis
        overdetection_plus1 = error_data[error_data['detection_error'] == 1]
        overdetection_plus2 = error_data[error_data['detection_error'] == 2]
        
        print("\nOverdetection analysis:")
        print(f"- Cases with +1 detection: {len(overdetection_plus1)}")
        print(f"- Cases with +2 detections: {len(overdetection_plus2)}")
        
        # Analyze per-bbox pixel ratios for overdetection cases
        self._analyze_overdetection_bbox_patterns(overdetection_plus1, "+1 Overdetection", outliers)
        self._analyze_overdetection_bbox_patterns(overdetection_plus2, "+2 Overdetection", outliers)
        
        # Standard large errors - BALANCED SAMPLING
        large_errors = error_data[error_data['abs_error'] >= 5]
        print(f"\nLarge errors (≥5): {len(large_errors)} total")
        
        # Sample from each model to ensure balanced representation
        for model in self.filtered_df['model_short'].unique():
            model_errors = large_errors[large_errors['model_short'] == model]
            print(f"  {model}: {len(model_errors)} large errors")
            
            # Take worst 3 from each model
            worst_for_model = model_errors.nlargest(3, 'abs_error')
            for _, row in worst_for_model.iterrows():
                bbox_analysis = self._extract_bbox_metrics(row)
                outliers.append({
                    'type': 'Large Count Error',
                    'severity': 'High', 
                    'description': f"Expected {int(row['expected_count'])}, detected {int(row['detected_count'])} ({row['detection_error']:+.0f})",
                    'model': row['model_short'],
                    'object': row['target_object'],
                    'variant': row['variant'],
                    'image_path': row['image_path'],
                    'expected_count': row['expected_count'],
                    'detected_count': row['detected_count'],
                    'error': row['detection_error'],
                    'pixel_ratio': row.get('pixel_ratio', 'N/A'),
                    'coverage_percentage': row.get('coverage_percentage', 'N/A'),
                    **bbox_analysis
                })
        
        # 2. Extreme relative errors - BALANCED SAMPLING
        extreme_relative_errors = error_data[
            (error_data['expected_count'] > 0) & 
            (error_data['abs_error'] / error_data['expected_count'] >= 0.8)
        ]
        
        print(f"\nExtreme relative errors (≥80%): {len(extreme_relative_errors)} total")
        
        # Sample relative errors per model
        for model in self.filtered_df['model_short'].unique():
            model_rel_errors = extreme_relative_errors[extreme_relative_errors['model_short'] == model]
            worst_rel_for_model = model_rel_errors.nlargest(2, 'abs_error')
            for _, row in worst_rel_for_model.iterrows():
                bbox_analysis = self._extract_bbox_metrics(row)
                outliers.append({
                    'type': 'Extreme Relative Error',
                    'severity': 'High',
                    'description': f"Expected {int(row['expected_count'])}, detected {int(row['detected_count'])} ({row['abs_error']/row['expected_count']*100:.0f}% error)",
                    'model': row['model_short'],
                    'object': row['target_object'],
                    'variant': row['variant'],
                    'image_path': row['image_path'],
                    'expected_count': row['expected_count'],
                    'detected_count': row['detected_count'],
                    'error': row['detection_error'],
                    'pixel_ratio': row.get('pixel_ratio', 'N/A'),
                    'coverage_percentage': row.get('coverage_percentage', 'N/A'),
                    **bbox_analysis
                })
    
        # 3. Extreme pixel ratios with bbox breakdown
        pixel_data = self.filtered_df.dropna(subset=['pixel_ratio', 'coverage_percentage'])
        
        if len(pixel_data) > 0:
            zero_pixels = pixel_data[pixel_data['pixel_ratio'] <= 0.001]
            extreme_high_pixels = pixel_data[pixel_data['coverage_percentage'] >= 90]
            
            print("\nExtreme pixel coverage:")
            print(f"- Zero pixels (≤0.1%): {len(zero_pixels)} total")
            print(f"- High pixels (≥90%): {len(extreme_high_pixels)} total")
            
            # Sample zero pixel cases per model with bbox analysis
            for model in self.filtered_df['model_short'].unique():
                model_zero = zero_pixels[zero_pixels['model_short'] == model]
                for _, row in model_zero.head(2).iterrows():
                    bbox_analysis = self._extract_bbox_metrics(row)
                    outliers.append({
                        'type': 'Zero Pixel Coverage',
                        'severity': 'Medium',
                        'description': f"Pixel ratio: {row['pixel_ratio']:.4f} ({row['coverage_percentage']:.1f}%)",
                        'model': row['model_short'],
                        'object': row['target_object'],
                        'variant': row['variant'],
                        'image_path': row['image_path'],
                        'expected_count': row['expected_count'],
                        'detected_count': row['detected_count'],
                        'error': row['detected_count'] - row['expected_count'],
                        'pixel_ratio': row['pixel_ratio'],
                        'coverage_percentage': row['coverage_percentage'],
                        **bbox_analysis
                    })
            
            # Sample extreme high pixel cases per model with bbox analysis
            for model in self.filtered_df['model_short'].unique():
                model_high = extreme_high_pixels[extreme_high_pixels['model_short'] == model]
                model_high_sorted = model_high.nlargest(2, 'pixel_ratio')
                for _, row in model_high_sorted.iterrows():
                    bbox_analysis = self._extract_bbox_metrics(row)
                    outliers.append({
                        'type': 'Extreme High Pixel Coverage',
                        'severity': 'Medium',
                        'description': f"Pixel ratio: {row['pixel_ratio']:.4f} ({row['coverage_percentage']:.1f}%)",
                        'model': row['model_short'],
                        'object': row['target_object'],
                        'variant': row['variant'],
                        'image_path': row['image_path'],
                        'expected_count': row['expected_count'],
                        'detected_count': row['detected_count'],
                        'error': row['detected_count'] - row['expected_count'],
                        'pixel_ratio': row['pixel_ratio'],
                        'coverage_percentage': row['coverage_percentage'],
                        **bbox_analysis
                    })
        
        # 4. Perfect accuracy with high expected counts
        perfect_high_count = self.filtered_df[
            (self.filtered_df['count_accuracy'] == 1.0) & 
            (self.filtered_df['expected_count'] >= 8)
        ]
        
        print(f"\nPerfect high count cases (≥8 objects): {len(perfect_high_count)} total")
        
        # Sample perfect cases per model
        for model in self.filtered_df['model_short'].unique():
            model_perfect = perfect_high_count[perfect_high_count['model_short'] == model]
            model_perfect_sorted = model_perfect.nlargest(3, 'expected_count')
            for _, row in model_perfect_sorted.iterrows():
                bbox_analysis = self._extract_bbox_metrics(row)
                outliers.append({
                    'type': 'Perfect High Count',
                    'severity': 'Positive',
                    'description': f"Perfect accuracy for {int(row['expected_count'])} {row['target_object']}s",
                    'model': row['model_short'],
                    'object': row['target_object'],
                    'variant': row['variant'],
                    'image_path': row['image_path'],
                    'expected_count': row['expected_count'],
                    'detected_count': row['detected_count'],
                    'error': 0,
                    'pixel_ratio': row.get('pixel_ratio', 'N/A'),
                    'coverage_percentage': row.get('coverage_percentage', 'N/A'),
                    **bbox_analysis
                })
        
        # 5. Worst case for each model (most extreme error)
        print("\nWorst case for each model:")
        for model in self.filtered_df['model_short'].unique():
            model_data = error_data[error_data['model_short'] == model]
            if len(model_data) > 0:
                worst_case = model_data.loc[model_data['abs_error'].idxmax()]
                print(f"  {model}: Expected {int(worst_case['expected_count'])}, detected {int(worst_case['detected_count'])} (error: {worst_case['detection_error']:+.0f})")
                
                bbox_analysis = self._extract_bbox_metrics(worst_case)
                outliers.append({
                    'type': f'Worst Case for {model}',
                    'severity': 'Critical',
                    'description': f"Worst failure: Expected {int(worst_case['expected_count'])}, detected {int(worst_case['detected_count'])} ({worst_case['detection_error']:+.0f})",
                    'model': model,
                    'object': worst_case['target_object'],
                    'variant': worst_case['variant'],
                    'image_path': worst_case['image_path'],
                    'expected_count': worst_case['expected_count'],
                    'detected_count': worst_case['detected_count'],
                    'error': worst_case['detection_error'],
                    'pixel_ratio': worst_case.get('pixel_ratio', 'N/A'),
                    'coverage_percentage': worst_case.get('coverage_percentage', 'N/A'),
                    **bbox_analysis
                })
        
        # Print detailed outlier information with bbox analysis
        print("\n=== DETAILED OUTLIER REPORT WITH BBOX ANALYSIS ===")
        outliers_df = pd.DataFrame(outliers)
        
        for model in sorted(self.filtered_df['model_short'].unique()):
            model_outliers = outliers_df[outliers_df['model'] == model]
            print(f"\n--- {model} ({len(model_outliers)} outliers) ---")
            
            for i, (_, outlier) in enumerate(model_outliers.iterrows(), 1):
                print(f"{i}. {outlier['type']} [{outlier['severity']}]")
                print(f"   {outlier['description']}")
                print(f"   Object: {outlier['object']}, Variant: {outlier['variant']}")
                print(f"   Expected: {outlier['expected_count']}, Detected: {outlier['detected_count']}")
                if outlier['pixel_ratio'] != 'N/A':
                    print(f"   Pixel ratio: {outlier['pixel_ratio']:.4f}")
                
                # Print bbox analysis if available
                if 'num_bboxes' in outlier and outlier['num_bboxes'] != 'N/A':
                    print(f"   BBox Analysis: {outlier['num_bboxes']} bboxes, ratios: {outlier['bbox_ratio_range']}")
                    if outlier.get('suspicious_bboxes', 'N/A') != 'N/A':
                        print(f"   Suspicious bboxes: {outlier['suspicious_bboxes']}")
                
                print(f"   Image: {outlier['image_path']}")
                print()
        
        # Save outliers to CSV
        outliers_file = self.output_dir / "outliers_and_extreme_cases_with_bbox.csv"
        outliers_df.to_csv(outliers_file, index=False)
        print(f"Enhanced outlier analysis with bbox data saved to: {outliers_file}")
        
        # Create enhanced visualization
        self._visualize_outliers_with_bbox(outliers_df, pixel_data if len(pixel_data) > 0 else None)
        
        return outliers_df
    
    def _visualize_outliers_with_bbox(self, outliers_df, pixel_data):
        """Create enhanced visualizations for outlier analysis including bbox metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Outlier types distribution
        if len(outliers_df) > 0:
            outlier_counts = outliers_df['type'].value_counts()
            axes[0, 0].bar(range(len(outlier_counts)), outlier_counts.values)
            axes[0, 0].set_xticks(range(len(outlier_counts)))
            axes[0, 0].set_xticklabels(outlier_counts.index, rotation=45, ha='right')
            axes[0, 0].set_title('Distribution of Outlier Types (Enhanced)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error magnitude distribution
        error_data = self.filtered_df.copy()
        error_data['abs_error'] = (error_data['detected_count'] - error_data['expected_count']).abs()
        
        axes[0, 1].hist(error_data['abs_error'], bins=range(0, int(error_data['abs_error'].max()) + 2), 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=1, color='orange', linestyle='--', label='+1 overdetection')
        axes[0, 1].axvline(x=2, color='red', linestyle='--', label='+2 overdetection')
        axes[0, 1].axvline(x=5, color='darkred', linestyle='--', label='Extreme error threshold')
        axes[0, 1].set_title('Distribution of Absolute Count Errors')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Pixel ratio distribution with outliers highlighted
        if pixel_data is not None:
            axes[0, 2].hist(pixel_data['pixel_ratio'], bins=50, alpha=0.7, color='blue', label='All data')
            
            # Highlight extreme cases
            zero_pixels = pixel_data[pixel_data['pixel_ratio'] <= 0.001]
            high_pixels = pixel_data[pixel_data['coverage_percentage'] >= 90]
            
            if len(zero_pixels) > 0:
                axes[0, 2].axvline(x=0.001, color='red', linestyle='--', label='Zero pixel threshold')
            if len(high_pixels) > 0:
                high_ratio_threshold = high_pixels['pixel_ratio'].min()
                axes[0, 2].axvline(x=high_ratio_threshold, color='orange', linestyle='--', 
                                 label='High coverage threshold')
            
            axes[0, 2].set_title('Pixel Ratio Distribution with Outlier Thresholds')
            axes[0, 2].set_xlabel('Pixel Ratio')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. BBox analysis: Number of bboxes vs error type
        bbox_outliers = outliers_df[outliers_df['num_bboxes'] != 'N/A'].copy()
        if len(bbox_outliers) > 0:
            # Convert num_bboxes to numeric
            bbox_outliers['num_bboxes_num'] = pd.to_numeric(bbox_outliers['num_bboxes'], errors='coerce')
            
            # Group by error type
            for error_type in ['+1 Overdetection', '+2 Overdetection']:
                search_term = error_type.split()[0].replace('+', r'\+')  # Escape the + sign for regex
                type_data = bbox_outliers[bbox_outliers['type'].str.contains(search_term, regex=True)]
                if len(type_data) > 0:
                    axes[1, 0].scatter(type_data['num_bboxes_num'], 
                                     [error_type] * len(type_data), 
                                     alpha=0.6, label=error_type)
            
            axes[1, 0].set_xlabel('Number of BBoxes Detected')
            axes[1, 0].set_ylabel('Error Type')
            axes[1, 0].set_title('BBoxes vs Overdetection Patterns')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Model performance variability
        model_std = self.filtered_df.groupby(['target_object', 'expected_count'])['count_accuracy'].agg(['mean', 'std']).reset_index()
        scatter = axes[1, 1].scatter(model_std['mean'], model_std['std'], alpha=0.6, c=model_std['expected_count'], cmap='viridis')
        axes[1, 1].set_xlabel('Mean Accuracy Across Models')
        axes[1, 1].set_ylabel('Standard Deviation Across Models')
        axes[1, 1].set_title('Model Agreement vs Performance')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Expected Count')
        
        # 6. BBox pixel ratio variance for overdetection cases
        if len(bbox_outliers) > 0:
            overdetection_types = bbox_outliers[bbox_outliers['type'].str.contains('Overdetection')]
            if len(overdetection_types) > 0:
                # Extract bbox ratio ranges
                ratio_variances = []
                for _, row in overdetection_types.iterrows():
                    if row['bbox_ratio_range'] != 'N/A':
                        try:
                            min_ratio, max_ratio = map(float, row['bbox_ratio_range'].split('-'))
                            variance = max_ratio - min_ratio
                            ratio_variances.append(variance)
                        except (ValueError, IndexError):
                            continue
                
                if ratio_variances:
                    axes[1, 2].hist(ratio_variances, bins=20, alpha=0.7, color='orange', 
                                   edgecolor='black')
                    axes[1, 2].set_title('BBox Pixel Ratio Variance in Overdetection')
                    axes[1, 2].set_xlabel('Pixel Ratio Range (Max - Min)')
                    axes[1, 2].set_ylabel('Frequency')
                    axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "enhanced_outlier_analysis_with_bbox.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced outlier visualization with bbox analysis saved to: {plot_path}")
        
    def _visualize_outliers(self, outliers_df, pixel_data):
        """Create visualizations for outlier analysis (legacy method for backward compatibility)."""
        # This method is kept for backward compatibility but now calls the enhanced version
        self._visualize_outliers_with_bbox(outliers_df, pixel_data)
    
    def _analyze_overdetection_bbox_patterns(self, overdetection_data, error_type, outliers_list):
        """Analyze per-bbox pixel ratio patterns for overdetection cases."""
        if len(overdetection_data) == 0:
            return
            
        print(f"\n--- {error_type} BBox Pattern Analysis ---")
        
        bbox_insights = []
        
        for _, row in overdetection_data.iterrows():
            if pd.isna(row.get('bbox_pixel_ratios')):
                continue
                
            try:
                bbox_data = json.loads(row['bbox_pixel_ratios'])
                if not bbox_data:
                    continue
                
                # Analyze bbox pixel ratios
                pixel_ratios = [bbox['pixel_ratio'] for bbox in bbox_data]
                confidences = [bbox['confidence'] for bbox in bbox_data]
                
                # Identify suspicious patterns
                very_low_ratio = [r for r in pixel_ratios if r < 0.1]  # Less than 10% filled
                very_high_ratio = [r for r in pixel_ratios if r > 0.9]  # More than 90% filled
                low_confidence = [c for c in confidences if c < 0.5]  # Low confidence detections
                
                # Pattern analysis
                pattern_description = []
                if len(very_low_ratio) > 0:
                    pattern_description.append(f"{len(very_low_ratio)} bbox(es) with <10% pixel fill")
                if len(very_high_ratio) > 0:
                    pattern_description.append(f"{len(very_high_ratio)} bbox(es) with >90% pixel fill")
                if len(low_confidence) > 0:
                    pattern_description.append(f"{len(low_confidence)} low confidence detection(s)")
                
                bbox_insights.append({
                    'image_path': row['image_path'],
                    'model': row['model_short'],
                    'object': row['target_object'],
                    'num_bboxes': len(bbox_data),
                    'pixel_ratios': pixel_ratios,
                    'avg_confidence': np.mean(confidences),
                    'pattern': "; ".join(pattern_description) if pattern_description else "Normal pattern",
                    'detection_error': row['detection_error']
                })
                
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Print summary of bbox patterns
        if bbox_insights:
            print(f"Analyzed {len(bbox_insights)} cases with bbox data:")
            
            # Count patterns
            very_low_count = sum(1 for insight in bbox_insights if "10% pixel fill" in insight['pattern'])
            very_high_count = sum(1 for insight in bbox_insights if "90% pixel fill" in insight['pattern'])
            low_conf_count = sum(1 for insight in bbox_insights if "low confidence" in insight['pattern'])
            
            print(f"  - Cases with very low pixel fill bboxes: {very_low_count}")
            print(f"  - Cases with very high pixel fill bboxes: {very_high_count}")
            print(f"  - Cases with low confidence detections: {low_conf_count}")
            
            # Sample most interesting cases for each model
            for model in self.filtered_df['model_short'].unique():
                model_insights = [insight for insight in bbox_insights if insight['model'] == model]
                if not model_insights:
                    continue
                
                # Sort by most suspicious patterns (most low pixel ratios + low confidence)
                model_insights.sort(key=lambda x: (
                    x['pattern'].count('10% pixel fill') + 
                    x['pattern'].count('low confidence')
                ), reverse=True)
                
                # Take top 2 most suspicious cases
                for insight in model_insights[:2]:
                    bbox_analysis = {
                        'num_bboxes': insight['num_bboxes'],
                        'bbox_ratio_range': f"{min(insight['pixel_ratios']):.3f}-{max(insight['pixel_ratios']):.3f}",
                        'avg_bbox_confidence': f"{insight['avg_confidence']:.3f}",
                        'suspicious_bboxes': insight['pattern']
                    }
                    
                    outliers_list.append({
                        'type': f'{error_type} BBox Pattern',
                        'severity': 'Medium',
                        'description': f"Detected {insight['detection_error']:+d} extra, bbox pattern: {insight['pattern']}",
                        'model': insight['model'],
                        'object': insight['object'],
                        'variant': 'N/A',  # Not available in this context
                        'image_path': insight['image_path'],
                        'expected_count': 'N/A',
                        'detected_count': 'N/A',
                        'error': insight['detection_error'],
                        'pixel_ratio': 'N/A',
                        'coverage_percentage': 'N/A',
                        **bbox_analysis
                    })
    
    def _extract_bbox_metrics(self, row):
        """Extract bbox metrics from a row for outlier analysis."""
        bbox_analysis = {
            'num_bboxes': 'N/A',
            'bbox_ratio_range': 'N/A',
            'avg_bbox_confidence': 'N/A',
            'suspicious_bboxes': 'N/A'
        }
        
        if pd.notna(row.get('bbox_pixel_ratios')):
            try:
                bbox_data = json.loads(row['bbox_pixel_ratios'])
                if bbox_data:
                    pixel_ratios = [bbox['pixel_ratio'] for bbox in bbox_data]
                    confidences = [bbox['confidence'] for bbox in bbox_data]
                    
                    bbox_analysis['num_bboxes'] = len(bbox_data)
                    bbox_analysis['bbox_ratio_range'] = f"{min(pixel_ratios):.3f}-{max(pixel_ratios):.3f}"
                    bbox_analysis['avg_bbox_confidence'] = f"{np.mean(confidences):.3f}"
                    
                    # Identify suspicious patterns
                    suspicious = []
                    if any(r < 0.1 for r in pixel_ratios):
                        suspicious.append("very low fill")
                    if any(r > 0.9 for r in pixel_ratios):
                        suspicious.append("very high fill") 
                    if any(c < 0.5 for c in confidences):
                        suspicious.append("low confidence")
                    
                    bbox_analysis['suspicious_bboxes'] = "; ".join(suspicious) if suspicious else "normal"
                    
            except (json.JSONDecodeError, KeyError):
                pass
        
        return bbox_analysis
    
    def _visualize_outliers(self, outliers_df, pixel_data):
        """Create visualizations for outlier analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Outlier types distribution
        if len(outliers_df) > 0:
            outlier_counts = outliers_df['type'].value_counts()
            axes[0, 0].bar(range(len(outlier_counts)), outlier_counts.values)
            axes[0, 0].set_xticks(range(len(outlier_counts)))
            axes[0, 0].set_xticklabels(outlier_counts.index, rotation=45, ha='right')
            axes[0, 0].set_title('Distribution of Outlier Types')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error magnitude distribution
        error_data = self.filtered_df.copy()
        error_data['abs_error'] = (error_data['detected_count'] - error_data['expected_count']).abs()
        
        axes[0, 1].hist(error_data['abs_error'], bins=range(0, int(error_data['abs_error'].max()) + 2), 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=5, color='red', linestyle='--', label='Extreme error threshold')
        axes[0, 1].set_title('Distribution of Absolute Count Errors')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Pixel ratio distribution with outliers highlighted
        if pixel_data is not None:
            axes[1, 0].hist(pixel_data['pixel_ratio'], bins=50, alpha=0.7, color='blue', label='All data')
            
            # Highlight extreme cases
            zero_pixels = pixel_data[pixel_data['pixel_ratio'] <= 0.001]
            high_pixels = pixel_data[pixel_data['coverage_percentage'] >= 90]
            
            if len(zero_pixels) > 0:
                axes[1, 0].axvline(x=0.001, color='red', linestyle='--', label='Zero pixel threshold')
            if len(high_pixels) > 0:
                high_ratio_threshold = high_pixels['pixel_ratio'].min()
                axes[1, 0].axvline(x=high_ratio_threshold, color='orange', linestyle='--', 
                                 label='High coverage threshold')
            
            axes[1, 0].set_title('Pixel Ratio Distribution with Outlier Thresholds')
            axes[1, 0].set_xlabel('Pixel Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model performance variability
        model_std = self.filtered_df.groupby(['target_object', 'expected_count'])['count_accuracy'].agg(['mean', 'std']).reset_index()
        scatter = axes[1, 1].scatter(model_std['mean'], model_std['std'], alpha=0.6, c=model_std['expected_count'], cmap='viridis')
        axes[1, 1].set_xlabel('Mean Accuracy Across Models')
        axes[1, 1].set_ylabel('Standard Deviation Across Models')
        axes[1, 1].set_title('Model Agreement vs Performance')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Expected Count')
        
        plt.tight_layout()
        plot_path = self.output_dir / "outlier_analysis_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Outlier visualization saved to: {plot_path}")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n=== Generating Summary Report ===")
        
        report_lines = []
        report_lines.append("# Advanced Count Adherence Analysis Report")
        report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Images Analyzed: {len(self.filtered_df)}")
        report_lines.append("")
        
        # Overall statistics
        overall_accuracy = self.filtered_df['count_accuracy'].mean()
        report_lines.append("## Overall Performance")
        report_lines.append(f"- Average Count Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        report_lines.append(f"- Standard Deviation: {self.filtered_df['count_accuracy'].std():.3f}")
        report_lines.append("")
        
        # Model performance
        model_performance = self.filtered_df.groupby('model_short')['count_accuracy'].agg(['mean', 'std', 'count'])
        report_lines.append("## Model Performance Summary")
        for model, model_stats in model_performance.iterrows():
            report_lines.append(f"- {model}: {model_stats['mean']:.3f} ± {model_stats['std']:.3f} (n={model_stats['count']})")
        report_lines.append("")
        
        # Object performance
        object_performance = self.filtered_df.groupby('target_object')['count_accuracy'].agg(['mean', 'std'])
        report_lines.append("## Object Performance Summary")
        for obj, obj_stats in object_performance.iterrows():
            report_lines.append(f"- {obj}: {obj_stats['mean']:.3f} ± {obj_stats['std']:.3f}")
        report_lines.append("")
        
        # Variant performance
        variant_performance = self.filtered_df.groupby('variant')['count_accuracy'].agg(['mean', 'std'])
        report_lines.append("## Variant Performance Summary")
        for variant, variant_stats in variant_performance.iterrows():
            report_lines.append(f"- {variant}: {variant_stats['mean']:.3f} ± {variant_stats['std']:.3f}")
        report_lines.append("")
        
        # Key findings
        report_lines.append("## Key Findings")
        
        # Best and worst models
        best_model = model_performance.loc[model_performance['mean'].idxmax()]
        worst_model = model_performance.loc[model_performance['mean'].idxmin()]
        report_lines.append(f"- Best performing model: {best_model.name} ({best_model['mean']:.3f})")
        report_lines.append(f"- Worst performing model: {worst_model.name} ({worst_model['mean']:.3f})")
        
        # Best and worst objects
        best_object = object_performance.loc[object_performance['mean'].idxmax()]
        worst_object = object_performance.loc[object_performance['mean'].idxmin()]
        report_lines.append(f"- Easiest object: {best_object.name} ({best_object['mean']:.3f})")
        report_lines.append(f"- Hardest object: {worst_object.name} ({worst_object['mean']:.3f})")
        
        # Performance by count
        count_performance = self.filtered_df.groupby('expected_count')['count_accuracy'].mean()
        best_count = count_performance.idxmax()
        worst_count = count_performance.idxmin()
        report_lines.append(f"- Best count performance: {best_count} objects ({count_performance[best_count]:.3f})")
        report_lines.append(f"- Worst count performance: {worst_count} objects ({count_performance[worst_count]:.3f})")
        
        # Pixel ratio insights
        if self.filtered_df['pixel_ratio'].notna().any():
            accurate_pixel_ratio = self.filtered_df[self.filtered_df['count_accuracy'] == 1]['pixel_ratio'].mean()
            inaccurate_pixel_ratio = self.filtered_df[self.filtered_df['count_accuracy'] == 0]['pixel_ratio'].mean()
            report_lines.append(f"- Average pixel ratio for accurate generations: {accurate_pixel_ratio:.4f}")
            report_lines.append(f"- Average pixel ratio for inaccurate generations: {inaccurate_pixel_ratio:.4f}")
        
        report_lines.append("")
        report_lines.append("## Files Generated")
        report_lines.append("- accuracy_by_count_object_model.csv: Detailed accuracy statistics")
        report_lines.append("- detailed_pixel_ratio_analysis.csv: Pixel ratio analysis by model and object")
        report_lines.append("- performance_degradation_analysis.csv: Analysis of performance vs count correlation")
        report_lines.append("- variant_performance_stats.csv: Performance comparison across variants")
        report_lines.append("- object_difficulty_analysis.csv: Object difficulty ranking")
        report_lines.append("- count_detection_error_stats.csv: Detection error patterns")
        report_lines.append("- model_consistency_analysis.csv: Model consistency metrics")
        report_lines.append("- correlation_matrix.csv: Correlation between different metrics")
        report_lines.append("- outliers_and_extreme_cases.csv: Detailed outlier and extreme case analysis")
        
        # Save report
        report_file = self.output_dir / "analysis_summary_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_file}")
        
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive count adherence analysis...")
        
        self.load_data()
        self.analyze_accuracy_by_count_and_object()
        self.analyze_pixel_ratio_accuracy_correlation()
        self.advanced_insights_analysis()
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Advanced Count Adherence Analysis for Diffusion Models")
    parser.add_argument("--data_folder", type=str, required=True,
                       help="Path to the data folder containing metrics.db")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Construct path to metrics.db
    data_path = Path(args.data_folder)
    db_path = data_path / "metrics.db"
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return
    
    # Create output directory
    output_dir = data_path / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Run analysis
    analyzer = CountAnalyzer(str(db_path), str(output_dir))
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
