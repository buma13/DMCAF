#!/usr/bin/env python3
"""
Hypothesis-Driven Count Adherence Analysis for Diffusion Models

This script provides categorical analysis organized around specific hypotheses about 
diffusion model count adherence performance. Each analysis section directly tests
one or more hypotheses with targeted metrics and visualizations.

Research Hypotheses:
1. Higher expected counts cause critical failure in all diffusion models
2. Numeral prompts perform worse than base prompts (word-based numbers)
3. Background inclusion reduces count adherence due to attention split
4. Model hierarchy exists but all converge to low accuracy at high counts
5. High pixel ratio + high confidence = correct detection indicator

Key Features:
- Hypothesis-driven organization
- Categorical data analysis
- Clear verification/debunking conclusions
- Practical insights over statistical complexity
- Focused on DMCAF framework specifics
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats

# Configure plotting for clarity
plt.style.use('default')
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (14, 10),
    'font.size': 12
})

# Constants for DMCAF framework
IGNORED_OBJECTS = {"brocolli"}  # Known problematic objects
YOLO_OBJECTS = {39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl"}
MODEL_COLORS = {
    'SD-v1-5': '#1f77b4',
    'SD-2-1': '#ff7f0e', 
    'SD-3-medium': '#2ca02c',
    'SD-3.5-medium': '#d62728'
}

# Critical thresholds based on DMCAF experience
CRITICAL_ACCURACY_THRESHOLD = 0.3  # Below this = critical failure
HIGH_PIXEL_RATIO_THRESHOLD = 0.8   # Above this = good segmentation
HIGH_CONFIDENCE_THRESHOLD = 0.8    # Above this = confident detection

class HypothesisAnalyzer:
    """Hypothesis-driven analyzer for count adherence in diffusion models."""
    
    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.df = None
        self.hypothesis_results = {}
        
    def load_data(self):
        """Load and preprocess data from SQLite database."""
        print("🔄 Loading DMCAF evaluation data...")
        conn = sqlite3.connect(self.db_path)
        
        # Load comprehensive evaluation data
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
        
        # Clean and prepare data
        self.df['model_short'] = (
            self.df['model_name']
            .str.split('/').str[-1]
            .str.replace('stable-diffusion-', 'SD-')
            .str.replace('-diffusers', '')
        )
        
        # Filter invalid data
        self.df = self.df[
            (self.df['count_accuracy'] >= 0) &
            (self.df['count_accuracy'] <= 1) &
            (~self.df['target_object'].isin(IGNORED_OBJECTS))
        ].copy()
        
        # Extract confidence data from bbox_pixel_ratios JSON
        self.df['has_high_confidence_objects'] = self.df['bbox_pixel_ratios'].apply(self._extract_confidence_info)
        
        print(f"✅ Loaded {len(self.df)} evaluations")
        print(f"📊 Models: {sorted(self.df['model_short'].unique())}")
        print(f"🎯 Objects: {sorted(self.df['target_object'].unique())}")
        print(f"🔄 Variants: {sorted(self.df['variant'].unique())}")
        print(f"🔢 Count range: {self.df['expected_count'].min()}-{self.df['expected_count'].max()}")
        
        # Check for potential sorting issues in database
        self._check_data_distribution()
        
    def _check_data_distribution(self):
        """Check if data is sorted by model (potential issue for sampling)."""
        # Check if data appears to be sorted by model
        model_positions = []
        for model in self.df['model_short'].unique():
            model_indices = self.df[self.df['model_short'] == model].index
            model_positions.append({
                'model': model,
                'first_index': model_indices.min(),
                'last_index': model_indices.max(),
                'count': len(model_indices)
            })
        
        # Sort by first appearance
        model_positions.sort(key=lambda x: x['first_index'])
        
        # Check if models appear in blocks (indicating sorted data)
        is_sorted_by_model = True
        for i in range(len(model_positions) - 1):
            if model_positions[i]['last_index'] >= model_positions[i+1]['first_index']:
                is_sorted_by_model = False
                break
                
        if is_sorted_by_model:
            print("⚠️  Database appears to be sorted by model - this may affect sampling")
            print("💡 Model order:", [mp['model'] for mp in model_positions])
        
    def _get_balanced_sample(self, df, target_variants, samples_per_combination=50):
        """Get a balanced sample across models and variants."""
        balanced_data = []
        
        for model in df['model_short'].unique():
            for variant in target_variants:
                model_variant_data = df[
                    (df['model_short'] == model) & 
                    (df['variant'] == variant)
                ]
                
                if len(model_variant_data) > 0:
                    # Sample up to target amount, or all if less available
                    sample_size = min(samples_per_combination, len(model_variant_data))
                    sample = model_variant_data.sample(n=sample_size, random_state=42)
                    balanced_data.append(sample)
        
        return pd.concat(balanced_data, ignore_index=True) if balanced_data else pd.DataFrame()
        
    def _extract_confidence_info(self, bbox_json_str):
        """Extract confidence information from bbox JSON data."""
        if pd.isna(bbox_json_str):
            return False
        try:
            bbox_data = json.loads(bbox_json_str)
            if isinstance(bbox_data, list) and len(bbox_data) > 0:
                high_conf_count = sum(1 for bbox in bbox_data if bbox.get('confidence', 0) >= HIGH_CONFIDENCE_THRESHOLD)
                return high_conf_count > 0
            return False
        except (json.JSONDecodeError, TypeError):
            return False
    
    def analyze_hypothesis_1_count_degradation(self):
        """
        HYPOTHESIS 1: Higher expected counts cause critical failure in all diffusion models
        
        Method: Analyze accuracy vs expected count patterns across all models
        """
        print("\n" + "="*80)
        print("🔬 HYPOTHESIS 1: Count Degradation Analysis")
        print("="*80)
        
        # Group by model and expected count
        grouped = self.df.groupby(['model_short', 'expected_count']).agg({
            'count_accuracy': ['mean', 'std', 'count'],
            'pixel_ratio': 'mean'
        }).round(4)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
        grouped = grouped.reset_index()
        
        # Create degradation analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy vs Count by Model
        for model in sorted(self.df['model_short'].unique()):
            model_data = grouped[grouped['model_short'] == model]
            ax1.plot(model_data['expected_count'], model_data['count_accuracy_mean'], 
                    marker='o', linewidth=2, label=model, color=MODEL_COLORS.get(model, 'black'))
        
        ax1.axhline(y=CRITICAL_ACCURACY_THRESHOLD, color='red', linestyle='--', alpha=0.7, 
                   label=f'Critical Threshold ({CRITICAL_ACCURACY_THRESHOLD})')
        ax1.set_xlabel('Expected Count')
        ax1.set_ylabel('Mean Accuracy')
        ax1.set_title('Accuracy Degradation by Expected Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Critical failure analysis
        critical_failures = self.df.groupby(['model_short', 'expected_count']).apply(
            lambda x: (x['count_accuracy'] < CRITICAL_ACCURACY_THRESHOLD).mean(), include_groups=False
        ).reset_index(name='failure_rate')
        
        for model in sorted(self.df['model_short'].unique()):
            model_data = critical_failures[critical_failures['model_short'] == model]
            ax2.plot(model_data['expected_count'], model_data['failure_rate'], 
                    marker='s', linewidth=2, label=model, color=MODEL_COLORS.get(model, 'black'))
        
        ax2.set_xlabel('Expected Count')
        ax2.set_ylabel('Critical Failure Rate')
        ax2.set_title('Critical Failure Rate by Expected Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Degradation correlation analysis
        correlations = []
        for model in self.df['model_short'].unique():
            model_df = self.df[self.df['model_short'] == model]
            if len(model_df) > 5:  # Need sufficient data
                corr, p_val = stats.pearsonr(model_df['expected_count'], model_df['count_accuracy'])
                correlations.append({
                    'model': model,
                    'correlation': corr,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
        
        corr_df = pd.DataFrame(correlations)
        bars = ax3.bar(corr_df['model'], corr_df['correlation'], 
                      color=[MODEL_COLORS.get(m, 'gray') for m in corr_df['model']])
        
        # Color significant correlations differently
        for i, (bar, sig) in enumerate(zip(bars, corr_df['significant'])):
            if sig:
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
        
        ax3.set_ylabel('Correlation (Count vs Accuracy)')
        ax3.set_title('Count-Accuracy Correlation by Model')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Critical failure threshold identification
        threshold_analysis = []
        for count in sorted(self.df['expected_count'].unique()):
            count_data = self.df[self.df['expected_count'] == count]
            
            # Calculate failure rate at each count level
            total_evaluations = len(count_data)
            critical_failures = (count_data['count_accuracy'] < CRITICAL_ACCURACY_THRESHOLD).sum()
            failure_rate = critical_failures / total_evaluations if total_evaluations > 0 else 0
            
            # Calculate average accuracy across all models
            avg_accuracy = count_data['count_accuracy'].mean()
            
            threshold_analysis.append({
                'expected_count': count,
                'failure_rate': failure_rate,
                'avg_accuracy': avg_accuracy,
                'sample_size': total_evaluations,
                'critical_point': failure_rate >= 0.5  # 50%+ failure rate
            })

        threshold_df = pd.DataFrame(threshold_analysis)

        # Create dual-axis plot showing both failure rate and accuracy
        ax4_twin = ax4.twinx()

        # Failure rate bars
        bars = ax4.bar(threshold_df['expected_count'], threshold_df['failure_rate'], 
                       alpha=0.6, color='red', width=0.6, label='Critical Failure Rate')

        # Average accuracy line
        ax4_twin.plot(threshold_df['expected_count'], threshold_df['avg_accuracy'], 
                     'bo-', linewidth=3, markersize=8, color='blue', 
                     label='Average Accuracy')

        # Mark critical threshold points
        critical_counts = threshold_df[threshold_df['critical_point']]['expected_count'].tolist()
        if critical_counts:
            first_critical = min(critical_counts)
            ax4.axvline(x=first_critical, color='darkred', linestyle='--', linewidth=2,
                        alpha=0.8, label=f'Critical Threshold (Count {first_critical})')
            
            # Add annotation
            ax4.text(first_critical + 0.1, 0.8, 
                     f'≥50% failure\nstarts at {first_critical}', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                     fontsize=11, fontweight='bold')

        # Formatting
        ax4.set_xlabel('Expected Count')
        ax4.set_ylabel('Critical Failure Rate', color='red')
        ax4_twin.set_ylabel('Average Accuracy', color='blue')
        ax4.set_title('Critical Failure Threshold Analysis\n(Where does degradation become critical?)')

        # Add horizontal reference line for failure threshold
        ax4.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='50% Failure Line')

        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_1_count_degradation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Conclusion analysis
        all_models_degrade = all(corr_df['correlation'] < -0.5) if len(corr_df) > 0 else False
        
        # Calculate critical failure at high counts from threshold analysis
        high_count_critical_failure = any(threshold_df[threshold_df['expected_count'] >= 7]['critical_point']) if len(threshold_df) > 0 else False
        critical_failure_starts_at = min(critical_counts) if critical_counts else None
        
        result = {
            'verified': all_models_degrade and high_count_critical_failure,
            'correlations': corr_df.to_dict('records') if len(corr_df) > 0 else [],
            'threshold_analysis': threshold_df.to_dict('records') if len(threshold_df) > 0 else [],
            'critical_failure_starts_at': critical_failure_starts_at,
            'conclusion': f"{'✅ VERIFIED' if all_models_degrade and high_count_critical_failure else '❌ PARTIALLY VERIFIED'}: "
                         f"All models degrade, critical failure threshold: {critical_failure_starts_at}"
        }
        
        self.hypothesis_results['H1_count_degradation'] = result
        correlations_str = [f"{r['model']}: {r['correlation']:.3f}" for r in result['correlations']]
        print(f"📊 Count-Accuracy Correlations: {correlations_str}")
        print(f"📉 {result['conclusion']}")
        
        return result
    
    def analyze_hypothesis_2_numeral_vs_text(self):
        """
        HYPOTHESIS 2: Numeral prompts perform worse than base prompts (word-based numbers)
        
        Method: Compare accuracy between 'numeral' and 'base' variants
        """
        print("\n" + "="*80)
        print("🔬 HYPOTHESIS 2: Numeral vs Text Prompt Analysis")
        print("="*80)
        
        # Filter for base and numeral variants only
        initial_variant_data = self.df[self.df['variant'].isin(['base', 'numeral'])].copy()
        
        if len(initial_variant_data) == 0:
            print("❌ No data found for base/numeral comparison")
            return {'verified': False, 'reason': 'No data available'}
        
        # Check if we need balanced sampling
        data_dist = initial_variant_data.groupby(['variant', 'model_short']).size().reset_index(name='count')
        expected_combinations = len(self.df['model_short'].unique()) * 2  # 2 variants
        
        if len(data_dist) < expected_combinations:
            print("⚠️  Missing model-variant combinations detected")
            variant_data = initial_variant_data  # Use all available data
        else:
            # Check if distribution is very uneven
            min_samples = data_dist['count'].min()
            max_samples = data_dist['count'].max()
            
            if max_samples > min_samples * 5:  # More than 5x difference
                print("📊 Using balanced sampling due to uneven distribution")
                variant_data = self._get_balanced_sample(self.df, ['base', 'numeral'], samples_per_combination=100)
            else:
                variant_data = initial_variant_data
        
        # Debug information about data distribution
        print(f"📊 Available variants in full dataset: {sorted(self.df['variant'].unique())}")
        print("📊 Data distribution after filtering:")
        data_dist = variant_data.groupby(['variant', 'model_short']).size().reset_index(name='count')
        for _, row in data_dist.iterrows():
            print(f"   {row['variant']} + {row['model_short']}: {row['count']} samples")
        
        # Check if we have balanced data across models
        models_per_variant = variant_data.groupby('variant')['model_short'].nunique()
        print(f"📊 Models per variant: {dict(models_per_variant)}")
        
        # If we have unbalanced data (some models missing from some variants), sample strategically
        if len(data_dist) < len(self.df['model_short'].unique()) * 2:  # Should have model*variant combinations
            print("⚠️  Warning: Unbalanced model distribution across variants")
            print("💡 Consider checking if all models were run for all variants")
        
        # Statistical comparison - enhanced visualization for presentation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Statistical significance testing with p-values and effect sizes
        # Plot 1: Statistical significance testing with p-values and effect sizes
        model_comparisons = []
        for model in variant_data['model_short'].unique():
            model_df = variant_data[variant_data['model_short'] == model]
            
            base_acc = model_df[model_df['variant'] == 'base']['count_accuracy']
            numeral_acc = model_df[model_df['variant'] == 'numeral']['count_accuracy']
            
            if len(base_acc) > 0 and len(numeral_acc) > 0:
                # Welch's t-test (unequal variances)
                t_stat, p_val = stats.ttest_ind(base_acc, numeral_acc, equal_var=False)
                
                # Calculate Cohen's d
                if len(base_acc) > 1 and len(numeral_acc) > 1:
                    pooled_std = np.sqrt(((len(base_acc) - 1) * base_acc.std() ** 2 + 
                                        (len(numeral_acc) - 1) * numeral_acc.std() ** 2) / 
                                       (len(base_acc) + len(numeral_acc) - 2))
                    cohens_d = (base_acc.mean() - numeral_acc.mean()) / pooled_std if pooled_std > 0 else 0
                else:
                    cohens_d = 0
                
                model_comparisons.append({
                    'model': model,
                    'base_mean': base_acc.mean(),
                    'numeral_mean': numeral_acc.mean(),
                    'difference': base_acc.mean() - numeral_acc.mean(),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'significant': p_val < 0.05,
                    'base_better': base_acc.mean() > numeral_acc.mean()
                })
        
        comp_df = pd.DataFrame(model_comparisons)
        
        # Enhanced statistical visualization with p-values and effect sizes
        if len(comp_df) > 0:
            # Create a more informative statistical plot
            x_pos = np.arange(len(comp_df))
            
            # Primary bars for accuracy differences
            colors = ['#2E8B57' if diff > 0 else '#DC143C' for diff in comp_df['difference']]
            bars = ax1.bar(x_pos, comp_df['difference'], color=colors, alpha=0.8, width=0.6)
            
            # Add significance markers and effect size annotations
            for i, (bar, row) in enumerate(zip(bars, comp_df.itertuples())):
                # Mark statistical significance with border and better positioned stars
                if row.significant:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(3)
                    # Add significance star - better positioned to avoid title collision
                    height = bar.get_height()
                    star_text = '***' if row.p_value < 0.001 else '**' if row.p_value < 0.01 else '*'
                    y_offset = 0.005 if height > 0 else -0.005  # Smaller offset
                    ax1.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                            star_text, ha='center', va='bottom' if height > 0 else 'top', 
                            fontweight='bold', fontsize=12)  # Smaller font
                
                # Add effect size annotation - better positioned
                effect_size_text = f'd={row.cohens_d:.2f}'  # Reduced precision
                bar_height = bar.get_height()
                
                # Position effect size text inside bar if bar is tall enough, otherwise outside
                if abs(bar_height) > 0.03:
                    y_pos = bar_height/2
                    text_color = 'white'
                else:
                    y_pos = bar_height + (0.015 if bar_height > 0 else -0.015)
                    text_color = 'black'
                
                ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                        effect_size_text, ha='center', va='center', 
                        fontsize=9, fontweight='bold', color=text_color)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([model.replace('SD-', '') for model in comp_df['model']], rotation=45)
            ax1.set_ylabel('Accuracy Difference (Base - Numeral)')
            ax1.set_title('Statistical Significance: Base vs Numeral Prompts\n(*p<0.05, **p<0.01, ***p<0.001)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            
            # Set y-limits to accommodate annotations
            y_min, y_max = ax1.get_ylim()
            margin = (y_max - y_min) * 0.15  # 15% margin
            ax1.set_ylim(y_min - margin, y_max + margin)
            
            # Add effect size interpretation legend
            ax1.text(0.02, 0.98, 'Effect Size (Cohen\'s d):\nSmall: 0.2, Medium: 0.5, Large: 0.8', 
                    transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Plot 2: Independent t-test results with confidence intervals
        
        # Plot 2: Independent t-test results with confidence intervals
        if len(comp_df) > 0:
            # Calculate confidence intervals for the differences
            ci_data = []
            for model in variant_data['model_short'].unique():
                model_df = variant_data[variant_data['model_short'] == model]
                base_acc = model_df[model_df['variant'] == 'base']['count_accuracy']
                numeral_acc = model_df[model_df['variant'] == 'numeral']['count_accuracy']
                
                if len(base_acc) > 1 and len(numeral_acc) > 1:
                    # Bootstrap confidence interval for difference
                    n_bootstrap = 1000
                    bootstrap_diffs = []
                    for _ in range(n_bootstrap):
                        base_boot = np.random.choice(base_acc, size=len(base_acc), replace=True)
                        numeral_boot = np.random.choice(numeral_acc, size=len(numeral_acc), replace=True)
                        bootstrap_diffs.append(base_boot.mean() - numeral_boot.mean())
                    
                    ci_lower = np.percentile(bootstrap_diffs, 2.5)
                    ci_upper = np.percentile(bootstrap_diffs, 97.5)
                    
                    ci_data.append({
                        'model': model,
                        'mean_diff': base_acc.mean() - numeral_acc.mean(),
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        't_stat': comp_df[comp_df['model'] == model]['t_statistic'].iloc[0],
                        'p_value': comp_df[comp_df['model'] == model]['p_value'].iloc[0]
                    })
            
            if ci_data:
                ci_df = pd.DataFrame(ci_data)
                
                # Create confidence interval plot
                x_pos = np.arange(len(ci_df))
                
                # Plot confidence intervals as error bars
                colors = ['#2E8B57' if diff > 0 else '#DC143C' for diff in ci_df['mean_diff']]
                
                # Calculate error bar sizes
                yerr_lower = ci_df['mean_diff'] - ci_df['ci_lower']
                yerr_upper = ci_df['ci_upper'] - ci_df['mean_diff']
                yerr = [yerr_lower, yerr_upper]
                
                bars = ax2.bar(x_pos, ci_df['mean_diff'], color=colors, alpha=0.7, 
                              yerr=yerr, capsize=8, error_kw={'linewidth': 2, 'ecolor': 'black'})
                
                # Add t-statistic and p-value annotations with better positioning
                for i, (bar, row) in enumerate(zip(bars, ci_df.itertuples())):
                    # Calculate annotation positions with adaptive offsets
                    ci_range = row.ci_upper - row.ci_lower
                    upper_offset = max(0.005, ci_range * 0.1)  # Dynamic offset based on CI range
                    lower_offset = max(0.005, ci_range * 0.1)
                    
                    # Add t-statistic value above upper CI
                    ax2.text(bar.get_x() + bar.get_width()/2., 
                            row.ci_upper + upper_offset,
                            f't={row.t_stat:.1f}',  # Reduced precision
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    # Add p-value below lower CI
                    p_text = f'p={row.p_value:.2f}' if row.p_value >= 0.01 else 'p<0.01'  # Simplified p-value
                    ax2.text(bar.get_x() + bar.get_width()/2., 
                            row.ci_lower - lower_offset,
                            p_text, 
                            ha='center', va='top', fontsize=8, style='italic')
                
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([model.replace('SD-', '') for model in ci_df['model']], rotation=45)
                ax2.set_ylabel('Mean Difference with 95% CI')
                ax2.set_title('Welch\'s t-test Results: Base vs Numeral\n(Error bars show 95% confidence intervals)')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
                
                # Set y-limits to accommodate annotations
                y_min, y_max = ax2.get_ylim()
                margin = (y_max - y_min) * 0.2  # 20% margin for annotations
                ax2.set_ylim(y_min - margin, y_max + margin)
        
        # Plot 3: Performance by expected count (unchanged)
        # Plot 3: Performance by expected count (unchanged)
        count_variant_stats = variant_data.groupby(['expected_count', 'variant']).agg({
            'count_accuracy': 'mean'
        }).reset_index()
        
        pivot_data = count_variant_stats.pivot(index='expected_count', columns='variant', values='count_accuracy')
        
        if 'base' in pivot_data.columns and 'numeral' in pivot_data.columns:
            ax3.plot(pivot_data.index, pivot_data['base'], marker='o', linewidth=3, 
                    label='Base (text)', color='#2E8B57', markersize=8)
            ax3.plot(pivot_data.index, pivot_data['numeral'], marker='s', linewidth=3, 
                    label='Numeral', color='#DC143C', markersize=8)
            ax3.set_xlabel('Expected Count')
            ax3.set_ylabel('Mean Accuracy')
            ax3.set_title('Variant Performance by Expected Count')
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Highlight critical failure threshold
            ax3.axhline(y=CRITICAL_ACCURACY_THRESHOLD, color='red', linestyle=':', 
                       alpha=0.8, linewidth=2, label=f'Critical Threshold ({CRITICAL_ACCURACY_THRESHOLD})')
        
        # Plot 4: Effect size analysis with interpretation
        if len(comp_df) > 0:
            # Create effect size interpretation plot
            effect_sizes = comp_df['cohens_d'].values
            models = [model.replace('SD-', '') for model in comp_df['model']]
            
            # Color code by effect size magnitude
            colors = []
            for d in effect_sizes:
                abs_d = abs(d)
                if abs_d < 0.2:
                    colors.append('#808080')  # Gray for negligible
                elif abs_d < 0.5:
                    colors.append('#FFA500')  # Orange for small
                elif abs_d < 0.8:
                    colors.append('#FF4500')  # Red-orange for medium
                else:
                    colors.append('#8B0000')  # Dark red for large
            
            bars = ax4.barh(models, effect_sizes, color=colors, alpha=0.8)
            
            # Add effect size magnitude labels
            for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
                # Determine effect size category
                abs_d = abs(d)
                if abs_d < 0.2:
                    size_label = "Negligible"
                elif abs_d < 0.5:
                    size_label = "Small"
                elif abs_d < 0.8:
                    size_label = "Medium"
                else:
                    size_label = "Large"
                
                # Position text appropriately
                x_pos = d + (0.05 if d > 0 else -0.05)
                ax4.text(x_pos, i, f'{d:.3f}\n({size_label})', 
                        ha='left' if d > 0 else 'right', va='center', 
                        fontweight='bold', fontsize=10)
            
            ax4.set_xlabel('Cohen\'s d (Effect Size)')
            ax4.set_title('Effect Size Analysis: Base vs Numeral\n(Positive = Base Better)')
            ax4.grid(True, alpha=0.3, axis='x')
            ax4.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            
            # Add reference lines for effect size thresholds
            ax4.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small (0.2)')
            ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Medium (0.5)')
            ax4.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large (0.8)')
            ax4.axvline(x=-0.2, color='orange', linestyle='--', alpha=0.5)
            ax4.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5)
            ax4.axvline(x=-0.8, color='darkred', linestyle='--', alpha=0.5)
            
            # Add a subtle legend for thresholds
            ax4.text(0.98, 0.02, 'Effect Size Thresholds:\nSmall: ±0.2, Medium: ±0.5, Large: ±0.8', 
                    transform=ax4.transAxes, ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_2_numeral_vs_text.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Conclusion
        if len(comp_df) > 0:
            base_better_count = sum(comp_df['base_better'])
            significant_base_better = sum(comp_df['base_better'] & comp_df['significant'])
            hypothesis_verified = significant_base_better > len(comp_df) * 0.5  # Majority of models
        else:
            hypothesis_verified = False
            base_better_count = 0
            significant_base_better = 0
        
        result = {
            'verified': hypothesis_verified,
            'comparisons': comp_df.to_dict('records') if len(comp_df) > 0 else [],
            'base_better_count': base_better_count,
            'significant_base_better': significant_base_better,
            'total_models': len(comp_df) if len(comp_df) > 0 else 0,
            'conclusion': f"{'✅ VERIFIED' if hypothesis_verified else '❌ NOT VERIFIED'}: "
                         f"{significant_base_better}/{len(comp_df) if len(comp_df) > 0 else 0} models show significant base>numeral"
        }
        
        self.hypothesis_results['H2_numeral_vs_text'] = result
        print(f"📊 Base vs Numeral: {base_better_count}/{len(comp_df) if len(comp_df) > 0 else 0} models favor base")
        print(f"📈 {result['conclusion']}")
        
        return result
    
    def analyze_hypothesis_3_background_effect(self):
        """
        HYPOTHESIS 3: Background inclusion reduces count adherence due to attention split
        
        Method: Compare 'background' variant with 'base' variant with enhanced statistical visualization
        """
        print("\n" + "="*80)
        print("🔬 HYPOTHESIS 3: Background Effect Analysis")
        print("="*80)
        
        # Filter for base and background variants only
        initial_variant_data = self.df[self.df['variant'].isin(['base', 'background'])].copy()
        
        if len(initial_variant_data) == 0:
            print("❌ No data found for base/background comparison")
            return {'verified': False, 'reason': 'No data available'}
        
        # Check if we need balanced sampling
        data_dist = initial_variant_data.groupby(['variant', 'model_short']).size().reset_index(name='count')
        expected_combinations = len(self.df['model_short'].unique()) * 2  # 2 variants
        
        if len(data_dist) < expected_combinations:
            print("⚠️ Unbalanced data detected. Using strategic sampling...")
            variant_data = self._get_balanced_sample(initial_variant_data, ['base', 'background'])
        else:
            variant_data = initial_variant_data
        
        # Debug information about data distribution
        print(f"📊 Available variants in full dataset: {sorted(self.df['variant'].unique())}")
        print("📊 Data distribution after filtering:")
        data_dist = variant_data.groupby(['variant', 'model_short']).size().reset_index(name='count')
        for _, row in data_dist.iterrows():
            print(f"   {row['variant']}-{row['model_short']}: {row['count']} samples")
        
        # Check if we have balanced data across models
        models_per_variant = variant_data.groupby('variant')['model_short'].nunique()
        print(f"📊 Models per variant: {dict(models_per_variant)}")
        
        # If we have unbalanced data (some models missing from some variants), sample strategically
        if len(data_dist) < len(self.df['model_short'].unique()) * 2:
            print("⚠️ Some models missing from variants. Results may be limited to available combinations.")
        
        # Statistical comparison - enhanced visualization for presentation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Statistical significance testing with p-values and effect sizes
        model_comparisons = []
        for model in variant_data['model_short'].unique():
            model_df = variant_data[variant_data['model_short'] == model]
            
            base_acc = model_df[model_df['variant'] == 'base']['count_accuracy']
            bg_acc = model_df[model_df['variant'] == 'background']['count_accuracy']
            
            if len(base_acc) > 0 and len(bg_acc) > 0:
                # Welch's t-test (unequal variances)
                t_stat, p_val = stats.ttest_ind(base_acc, bg_acc, equal_var=False)
                
                # Calculate Cohen's d (effect size)
                pooled_std = np.sqrt((base_acc.var() + bg_acc.var()) / 2)
                cohens_d = (base_acc.mean() - bg_acc.mean()) / pooled_std if pooled_std > 0 else 0
                
                model_comparisons.append({
                    'model': model,
                    'base_mean': base_acc.mean(),
                    'background_mean': bg_acc.mean(),
                    'difference': base_acc.mean() - bg_acc.mean(),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'significant': p_val < 0.05,
                    'base_better': base_acc.mean() > bg_acc.mean()
                })
        
        comp_df = pd.DataFrame(model_comparisons)
        
        # Enhanced statistical visualization with p-values and effect sizes
        if len(comp_df) > 0:
            colors = ['#2E8B57' if diff > 0 else '#DC143C' for diff in comp_df['difference']]  # SeaGreen vs Crimson
            bars = ax1.bar(comp_df['model'], comp_df['difference'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Calculate adaptive y-axis margins for text positioning
            y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
            margin = y_range * 0.20  # Increased to 20% margin for annotations
            ax1.set_ylim(ax1.get_ylim()[0] - margin/2, ax1.get_ylim()[1] + margin)
            
            # Annotate bars with Cohen's d and significance
            for i, (bar, row) in enumerate(zip(bars, comp_df.itertuples())):
                # Significance stars
                if row.p_value < 0.001:
                    stars = "***"
                elif row.p_value < 0.01:
                    stars = "**"
                elif row.p_value < 0.05:
                    stars = "*"
                else:
                    stars = ""
                
                # Position Cohen's d based on bar height with better bounds checking
                bar_height = bar.get_height()
                y_limits = ax1.get_ylim()
                y_range_current = y_limits[1] - y_limits[0]
                
                # More conservative positioning to avoid protrusion
                if abs(bar_height) > y_range_current * 0.15:  # If bar is >15% of y-range, put inside
                    y_pos = bar_height * 0.5
                    text_color = 'white'
                else:  # Otherwise, put outside with conservative offset
                    offset = min(0.015, y_range_current * 0.08)  # Conservative offset
                    y_pos = bar_height + (offset if bar_height >= 0 else -offset)
                    text_color = 'black'
                    
                    # Double-check bounds to prevent protrusion
                    if y_pos > y_limits[1] * 0.95:  # Too close to top
                        y_pos = bar_height * 0.7
                        text_color = 'white'
                    elif y_pos < y_limits[0] * 0.95:  # Too close to bottom
                        y_pos = bar_height * 0.7
                        text_color = 'white'
                
                # Annotate Cohen's d with reduced precision
                ax1.text(bar.get_x() + bar.get_width()/2, y_pos, f'd={row.cohens_d:.2f}', 
                        ha='center', va='center', fontweight='bold', fontsize=11, color=text_color)
                
                # Add significance stars above/below bar with conservative positioning
                star_offset = min(0.02, y_range_current * 0.05)
                star_y_pos = bar_height + (star_offset if bar_height >= 0 else -star_offset)
                
                # Ensure stars don't protrude
                if star_y_pos > y_limits[1] * 0.98 or star_y_pos < y_limits[0] * 0.98:
                    star_y_pos = y_pos + (star_offset * 0.5 if bar_height >= 0 else -star_offset * 0.5)
                
                if stars:
                    ax1.text(bar.get_x() + bar.get_width()/2, star_y_pos, stars, 
                            ha='center', va='bottom' if bar_height >= 0 else 'top', 
                            fontweight='bold', fontsize=14, color='black')
            
            ax1.set_ylabel('Accuracy Difference (Base - Background)')
            ax1.set_title('Statistical Significance: Base vs Background\n(Positive = Base Better, with Effect Sizes)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Plot 2: Bootstrap confidence intervals with t-test results
        if len(comp_df) > 0:
            models = comp_df['model'].tolist()
            differences = comp_df['difference'].tolist()
            
            # Calculate bootstrap confidence intervals
            bootstrap_cis = []
            for model in models:
                model_df = variant_data[variant_data['model_short'] == model]
                base_acc = model_df[model_df['variant'] == 'base']['count_accuracy'].values
                bg_acc = model_df[model_df['variant'] == 'background']['count_accuracy'].values
                
                if len(base_acc) > 0 and len(bg_acc) > 0:
                    # Bootstrap sampling
                    bootstrap_diffs = []
                    for _ in range(1000):
                        base_sample = np.random.choice(base_acc, size=len(base_acc), replace=True)
                        bg_sample = np.random.choice(bg_acc, size=len(bg_acc), replace=True)
                        bootstrap_diffs.append(base_sample.mean() - bg_sample.mean())
                    
                    ci_lower = np.percentile(bootstrap_diffs, 2.5)
                    ci_upper = np.percentile(bootstrap_diffs, 97.5)
                    bootstrap_cis.append((ci_lower, ci_upper))
                else:
                    bootstrap_cis.append((0, 0))
            
            # Calculate error bars (distance from mean to CI bounds)
            ci_lower_errors = [diff - ci[0] for diff, ci in zip(differences, bootstrap_cis)]
            ci_upper_errors = [ci[1] - diff for diff, ci in zip(differences, bootstrap_cis)]
            
            bars = ax2.bar(models, differences, 
                          yerr=[ci_lower_errors, ci_upper_errors], 
                          capsize=8, alpha=0.8, 
                          color=['#2E8B57' if d > 0 else '#DC143C' for d in differences],
                          edgecolor='black', linewidth=1, error_kw={'linewidth': 2, 'ecolor': 'black'})
            
            # Calculate adaptive positioning for annotations
            y_range = max(ci_upper_errors) + max(differences) - min(differences) + max(ci_lower_errors)
            ci_range = y_range * 0.1  # Base offset on CI range
            ax2.set_ylim(min(differences) - max(ci_lower_errors) - y_range * 0.2, 
                        max(differences) + max(ci_upper_errors) + y_range * 0.2)
            
            # Annotate with t-statistics and p-values
            for i, (bar, row) in enumerate(zip(bars, comp_df.itertuples())):
                # Dynamic offset based on confidence interval
                offset = max(0.005, ci_range * 0.1)
                y_pos = differences[i] + ci_upper_errors[i] + offset
                
                ax2.text(bar.get_x() + bar.get_width()/2, y_pos, 
                        f't={row.t_statistic:.2f}\np={row.p_value:.3f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            ax2.set_ylabel('Accuracy Difference (Base - Background)')
            ax2.set_title('Bootstrap 95% Confidence Intervals\nwith Welch\'s t-test Results')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Plot 3: Performance by expected count (enhanced)
        count_variant_stats = variant_data.groupby(['expected_count', 'variant']).agg({
            'count_accuracy': 'mean'
        }).reset_index()
        
        pivot_data = count_variant_stats.pivot(index='expected_count', columns='variant', values='count_accuracy')
        
        if 'base' in pivot_data.columns and 'background' in pivot_data.columns:
            ax3.plot(pivot_data.index, pivot_data['base'], marker='o', linewidth=3, 
                    markersize=8, label='Base', color='#2E8B57', alpha=0.8)
            ax3.plot(pivot_data.index, pivot_data['background'], marker='s', linewidth=3, 
                    markersize=8, label='Background', color='#DC143C', alpha=0.8)
            
            ax3.axhline(y=CRITICAL_ACCURACY_THRESHOLD, color='red', linestyle='--', alpha=0.7, 
                       label=f'Critical Threshold ({CRITICAL_ACCURACY_THRESHOLD})', linewidth=2)
            ax3.set_xlabel('Expected Count')
            ax3.set_ylabel('Mean Accuracy')
            ax3.set_title('Performance by Expected Count: Base vs Background')
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Enhanced styling
            ax3.set_facecolor('#F8F8FF')  # Light background
        
        # Plot 4: Combined Coverage and Pixel Ratio Analysis (Simplified with Averages)
        # Calculate average metrics across all models for each variant
        avg_metrics = variant_data.groupby('variant').agg({
            'coverage_percentage': 'mean',
            'pixel_ratio': 'mean'
        }).reset_index()
        
        # Ensure we have both variants
        base_metrics = avg_metrics[avg_metrics['variant'] == 'base']
        bg_metrics = avg_metrics[avg_metrics['variant'] == 'background']
        
        if len(base_metrics) > 0 and len(bg_metrics) > 0:
            # Simple bar chart with dual y-axis for averages
            variants = ['Base', 'Background']
            x = np.arange(len(variants))
            width = 0.6
            
            # Coverage values (already in percentage)
            coverage_values = [
                base_metrics['coverage_percentage'].iloc[0],
                bg_metrics['coverage_percentage'].iloc[0]
            ]
            
            # Pixel ratio values converted to percentages
            pixel_values = [
                base_metrics['pixel_ratio'].iloc[0] * 100,  # Convert to percentage
                bg_metrics['pixel_ratio'].iloc[0] * 100     # Convert to percentage
            ]
            
            # Create twin axes for dual metrics
            ax4_twin = ax4.twinx()
            
            # Coverage bars (left y-axis)
            coverage_bars = ax4.bar(x, coverage_values, width, 
                                   color=['#4CAF50', '#FF6B6B'], alpha=0.8, 
                                   edgecolor='black', linewidth=2,
                                   label='Coverage Percentage')
            
            # Pixel ratio line with markers (right y-axis) - now in percentage
            ax4_twin.plot(x, pixel_values, 'ko-', linewidth=4, 
                         markersize=12, markerfacecolor='white',
                         markeredgewidth=3, markeredgecolor='black',
                         label='Pixel Ratio (%)')
            
            # Add threshold line for pixel ratio (convert to percentage)
            ax4_twin.axhline(y=HIGH_PIXEL_RATIO_THRESHOLD * 100, color='red', linestyle='--', 
                            alpha=0.7, linewidth=2, label=f'High Pixel Ratio ({HIGH_PIXEL_RATIO_THRESHOLD * 100:.0f}%)')
            
            # Add value annotations on bars
            for i, (bar, cov_val) in enumerate(zip(coverage_bars, coverage_values)):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(coverage_values) * 0.02,
                    f'{cov_val:.1f}%',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=12
                )

            # Add value annotations on pixel ratio points (already in percentage)
            for i, (x_pos, pixel_val) in enumerate(zip(x, pixel_values)):
                ax4_twin.text(
                    x_pos,
                    pixel_val + (max(pixel_values) - min(pixel_values)) * 0.05,
                    f'{pixel_val:.1f}%',  # pixel_val is already in percentage
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )

            # Force pixel ratio axis to start at 0 and go to 100
            ax4_twin.set_ylim(0, 100)
            ax4.set_ylim(0, 100)
            
            # Formatting
            ax4.set_xlabel('Prompt Variant', fontsize=14)
            ax4.set_ylabel('Coverage Percentage (%)', color='#2E7D32', fontsize=14)
            ax4_twin.set_ylabel('Pixel Ratio (%)', color='#8B0000', fontsize=14)
            ax4.set_title('Average Coverage & Pixel Ratio: Base vs Background\n(Both Metrics as Percentages, Averaged Across All Models)', fontsize=14)
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(variants, fontsize=13)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Color-code axes labels to match metrics
            ax4.tick_params(axis='y', labelcolor='#2E7D32', labelsize=12)
            ax4_twin.tick_params(axis='y', labelcolor='#8B0000', labelsize=12)
            
            # Combine legends with better positioning
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, 
                      loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for average comparison', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        
        # Color-code axes labels
        ax4.tick_params(axis='y', labelcolor='black')
        ax4_twin.tick_params(axis='y', labelcolor='black')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_3_background_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Conclusion
        if len(comp_df) > 0:
            base_better_count = sum(comp_df['base_better'])
            significant_base_better = sum(comp_df['base_better'] & comp_df['significant'])
            hypothesis_verified = significant_base_better > len(comp_df) * 0.5  # Majority of models
        else:
            hypothesis_verified = False
            base_better_count = 0
            significant_base_better = 0
        
        result = {
            'verified': hypothesis_verified,
            'comparisons': comp_df.to_dict('records') if len(comp_df) > 0 else [],
            'base_better_count': base_better_count,
            'significant_base_better': significant_base_better,
            'total_models': len(comp_df) if len(comp_df) > 0 else 0,
            'conclusion': f"{'✅ VERIFIED' if hypothesis_verified else '❌ NOT VERIFIED'}: "
                         f"{significant_base_better}/{len(comp_df) if len(comp_df) > 0 else 0} models show significant base>background"
        }
        
        self.hypothesis_results['H3_background_effect'] = result
        print(f"📊 Base vs Background: {base_better_count}/{len(comp_df) if len(comp_df) > 0 else 0} models favor base")
        print(f"🎯 {result['conclusion']}")
        
        return result
    
    def analyze_hypothesis_4_model_hierarchy(self):
        """
        HYPOTHESIS 4: Model hierarchy exists but all converge to low accuracy at high counts
        
        Method: Compare model performance overall and at high counts specifically
        """
        print("\n" + "="*80)
        print("🔬 HYPOTHESIS 4: Model Hierarchy Analysis")
        print("="*80)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Overall model performance ranking
        model_overall = self.df.groupby('model_short').agg({
            'count_accuracy': ['mean', 'std', 'count']
        }).round(4)
        model_overall.columns = ['_'.join(col).strip() for col in model_overall.columns]
        model_overall = model_overall.reset_index().sort_values('count_accuracy_mean', ascending=False)
        
        ax1.bar(model_overall['model_short'], model_overall['count_accuracy_mean'],
                      yerr=model_overall['count_accuracy_std'], capsize=5,
                      color=[MODEL_COLORS.get(m, 'gray') for m in model_overall['model_short']])
        ax1.set_ylabel('Mean Accuracy')
        ax1.set_title('Overall Model Performance Ranking')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model ranking stability across counts - shows when hierarchy collapses
        # Group by model and expected count to get mean accuracy for ranking
        grouped = self.df.groupby(['model_short', 'expected_count']).agg({
            'count_accuracy': 'mean'
        }).round(4).reset_index()
        
        ranking_data = []
        for count in grouped['expected_count'].unique():
            count_models = grouped[grouped['expected_count'] == count].sort_values('count_accuracy', ascending=False)
            for rank, (_, row) in enumerate(count_models.iterrows(), 1):
                ranking_data.append({
                    'expected_count': count,
                    'model': row['model_short'],
                    'rank': rank,
                    'accuracy': row['count_accuracy']
                })

        ranking_df = pd.DataFrame(ranking_data)

        # Create ranking stability plot
        for model in ranking_df['model'].unique():
            model_ranks = ranking_df[ranking_df['model'] == model]
            ax2.plot(model_ranks['expected_count'], model_ranks['rank'], 
                    marker='o', linewidth=3, markersize=8, label=model, 
                    color=MODEL_COLORS.get(model, 'black'), alpha=0.8)

        ax2.set_xlabel('Expected Count')
        ax2.set_ylabel('Model Rank (1 = Best)')
        ax2.set_title('Model Ranking Stability Across Counts\n(Shows when hierarchy collapses)')
        ax2.set_ylim(ax2.get_ylim()[::-1])  # Invert y-axis (rank 1 at top)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks(range(1, len(ranking_df['model'].unique()) + 1))

        # Calculate and annotate hierarchy stability
        stability_metrics = []
        for count in sorted(ranking_df['expected_count'].unique()):
            count_ranks = ranking_df[ranking_df['expected_count'] == count].set_index('model')['rank']
            
            # Calculate rank variance (low variance = stable hierarchy)
            rank_variance = count_ranks.var()
            
            # Check if all ranks are different (perfect hierarchy)
            perfect_hierarchy = len(set(count_ranks.values)) == len(count_ranks)
            
            stability_metrics.append({
                'count': count,
                'rank_variance': rank_variance,
                'perfect_hierarchy': perfect_hierarchy,
                'top_model': count_ranks.idxmin()  # Model with rank 1
            })

        # Find hierarchy collapse point (where rank variance becomes low or rankings shuffle significantly)
        stability_df = pd.DataFrame(stability_metrics)
        if len(stability_df) > 0:
            baseline_variance = stability_df['rank_variance'].iloc[0] if len(stability_df) > 0 else 0

            # Identify potential collapse points
            collapse_threshold = baseline_variance * 0.3  # 30% of initial variance
            collapse_points = stability_df[stability_df['rank_variance'] <= collapse_threshold]['count'].tolist()

            if collapse_points:
                first_collapse = min(collapse_points)
                ax2.axvline(x=first_collapse, color='red', linestyle='--', alpha=0.8, linewidth=2,
                           label=f'Hierarchy Instability (Count {first_collapse})')
                
                # Add annotation
                ax2.text(first_collapse + 0.1, len(ranking_df['model'].unique()) * 0.8, 
                        f'Hierarchy becomes\nunstable at count {first_collapse}', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=10)

            # Add ranking change annotations for significant shifts
            prev_top_model = None
            for i, row in stability_df.iterrows():
                if prev_top_model and prev_top_model != row['top_model']:
                    if prev_top_model and prev_top_model != row['top_model']:
                        # annotation removed
                        pass
                prev_top_model = row['top_model']

        # Update legend to include new elements
        ax2.legend(fontsize=10, loc='lower right')
        
        # Plot 3: Model performance across count ranges
        count_bins = [(1, 3), (4, 6), (7, 10)]
        bin_labels = ['Low (1-3)', 'Medium (4-6)', 'High (7-10)']
        
        model_by_bins = {}
        for model in self.df['model_short'].unique():
            model_df = self.df[self.df['model_short'] == model]
            bin_performance = []
            
            for low, high in count_bins:
                bin_data = model_df[(model_df['expected_count'] >= low) & (model_df['expected_count'] <= high)]
                bin_performance.append(bin_data['count_accuracy'].mean() if len(bin_data) > 0 else 0)
            
            model_by_bins[model] = bin_performance
        
        x = np.arange(len(bin_labels))
        width = 0.2
        
        for i, model in enumerate(sorted(model_by_bins.keys())):
            ax3.bar(x + i * width, model_by_bins[model], width, 
                   label=model, color=MODEL_COLORS.get(model, 'gray'))
        
        ax3.set_xlabel('Count Range')
        ax3.set_ylabel('Mean Accuracy')
        ax3.set_title('Model Performance by Count Range')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(bin_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Recalculate high count data for Plot 4 and conclusion analysis
        high_count_df = self.df[self.df['expected_count'] >= 7]
        model_high_count = high_count_df.groupby('model_short').agg({
            'count_accuracy': ['mean', 'std', 'count']
        }).round(4)
        model_high_count.columns = ['_'.join(col).strip() for col in model_high_count.columns]
        model_high_count = model_high_count.reset_index()
        
        # Plot 4: Convergence analysis - variance in high counts
        high_count_variance = high_count_df.groupby('model_short')['count_accuracy'].agg(['mean', 'std']).reset_index()
        
        ax4.bar(high_count_variance['model_short'], high_count_variance['std'],
                      color=[MODEL_COLORS.get(m, 'gray') for m in high_count_variance['model_short']])
        ax4.set_ylabel('Accuracy Standard Deviation')
        ax4.set_title('Performance Variance at High Counts')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_4_model_hierarchy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical analysis of hierarchy
        model_pairs = []
        models = list(self.df['model_short'].unique())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1_acc = self.df[self.df['model_short'] == models[i]]['count_accuracy']
                model2_acc = self.df[self.df['model_short'] == models[j]]['count_accuracy']
                
                t_stat, p_val = stats.ttest_ind(model1_acc, model2_acc, equal_var=False)
                
                model_pairs.append({
                    'model1': models[i],
                    'model2': models[j],
                    'model1_mean': model1_acc.mean(),
                    'model2_mean': model2_acc.mean(),
                    'difference': model1_acc.mean() - model2_acc.mean(),
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
        
        pairs_df = pd.DataFrame(model_pairs)
        
        # Check convergence at high counts and ranking stability
        all_low_at_high = (model_high_count['count_accuracy_mean'] < CRITICAL_ACCURACY_THRESHOLD).all()
        hierarchy_exists = len(pairs_df[pairs_df['significant']]) > 0
        
        # Add ranking stability analysis to results
        hierarchy_collapse_point = None
        stability_df = pd.DataFrame()  # Initialize in case not created above
        collapse_points = []  # Initialize in case not created above
        
        # Check if stability analysis was performed in Plot 2
        if 'stability_df' in locals() and len(stability_df) > 0 and 'collapse_points' in locals():
            if len(collapse_points) > 0:
                hierarchy_collapse_point = min(collapse_points)
        
        result = {
            'verified': hierarchy_exists and all_low_at_high,
            'overall_ranking': model_overall.to_dict('records'),
            'high_count_performance': model_high_count.to_dict('records'),
            'ranking_stability': stability_df.to_dict('records') if len(stability_df) > 0 else [],
            'hierarchy_collapse_point': hierarchy_collapse_point,
            'significant_pairs': len(pairs_df[pairs_df['significant']]),
            'total_pairs': len(pairs_df),
            'all_critical_at_high': all_low_at_high,
            'conclusion': f"{'✅ VERIFIED' if hierarchy_exists and all_low_at_high else '❌ PARTIALLY VERIFIED'}: "
                         f"Hierarchy exists: {hierarchy_exists}, Collapses at count: {hierarchy_collapse_point}, All critical at high counts: {all_low_at_high}"
        }
        
        self.hypothesis_results['H4_model_hierarchy'] = result
        print(f"📊 Model hierarchy: {len(pairs_df[pairs_df['significant']])}/{len(pairs_df)} significant differences")
        print(f"⚠️ All models critical at high counts: {all_low_at_high}")
        print(f"🏆 {result['conclusion']}")
        
        return result
    
    def analyze_hypothesis_5_pixel_confidence_relationship(self):
        """
        HYPOTHESIS 5: High pixel ratio + high confidence = correct detection
        
        Method: Analyze relationship between pixel ratio, confidence, and accuracy
        """
        print("\n" + "="*80)
        print("🔬 HYPOTHESIS 5: Pixel Ratio & Confidence Analysis")
        print("="*80)
        
        # Filter data with valid pixel ratios
        valid_data = self.df[self.df['pixel_ratio'].notna()].copy()
        
        if len(valid_data) == 0:
            print("❌ No valid pixel ratio data found")
            return {'verified': False, 'reason': 'No valid data'}
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Pixel ratio vs accuracy
        scatter = ax1.scatter(valid_data['pixel_ratio'], valid_data['count_accuracy'], 
                            alpha=0.6, c=valid_data['expected_count'], cmap='viridis', s=20)
        ax1.set_xlabel('Pixel Ratio (Segmented/BBox)')
        ax1.set_ylabel('Count Accuracy')
        ax1.set_title('Pixel Ratio vs Count Accuracy')
        ax1.axvline(x=HIGH_PIXEL_RATIO_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                   label=f'High Ratio Threshold ({HIGH_PIXEL_RATIO_THRESHOLD})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add correlation
        corr, p_val = stats.pearsonr(valid_data['pixel_ratio'], valid_data['count_accuracy'])
        ax1.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_val:.3f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.colorbar(scatter, ax=ax1, label='Expected Count')
        
        # Plot 2: High pixel ratio performance
        valid_data['high_pixel_ratio'] = valid_data['pixel_ratio'] >= HIGH_PIXEL_RATIO_THRESHOLD
        
        pixel_performance = valid_data.groupby(['high_pixel_ratio', 'model_short']).agg({
            'count_accuracy': 'mean'
        }).reset_index()
        
        sns.barplot(data=pixel_performance, x='model_short', y='count_accuracy', 
                   hue='high_pixel_ratio', ax=ax2)
        ax2.set_title('Performance: High vs Low Pixel Ratio')
        ax2.set_ylabel('Mean Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence analysis (from bbox data)
        conf_analysis = []
        for _, row in valid_data.iterrows():
            if pd.notna(row['bbox_pixel_ratios']):
                try:
                    bbox_data = json.loads(row['bbox_pixel_ratios'])
                    if isinstance(bbox_data, list) and len(bbox_data) > 0:
                        avg_confidence = np.mean([bbox.get('confidence', 0) for bbox in bbox_data])
                        high_conf_ratio = sum(1 for bbox in bbox_data if bbox.get('confidence', 0) >= HIGH_CONFIDENCE_THRESHOLD) / len(bbox_data)
                        
                        conf_analysis.append({
                            'count_accuracy': row['count_accuracy'],
                            'pixel_ratio': row['pixel_ratio'],
                            'avg_confidence': avg_confidence,
                            'high_conf_ratio': high_conf_ratio,
                            'model': row['model_short'],
                            'expected_count': row['expected_count']
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
        
        if conf_analysis:
            conf_df = pd.DataFrame(conf_analysis)
            
            scatter2 = ax3.scatter(conf_df['avg_confidence'], conf_df['count_accuracy'], 
                                 alpha=0.6, c=conf_df['pixel_ratio'], cmap='plasma', s=20)
            ax3.set_xlabel('Average YOLO Confidence')
            ax3.set_ylabel('Count Accuracy')
            ax3.set_title('Confidence vs Count Accuracy')
            ax3.axvline(x=HIGH_CONFIDENCE_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                       label='High Confidence Threshold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            plt.colorbar(scatter2, ax=ax3, label='Pixel Ratio')
            
            # Correlation
            conf_corr, conf_p = stats.pearsonr(conf_df['avg_confidence'], conf_df['count_accuracy'])
            ax3.text(0.05, 0.95, f'r = {conf_corr:.3f}, p = {conf_p:.3f}', 
                    transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        # Plot 4: Combined high pixel ratio + high confidence analysis
        if conf_analysis:
            conf_df['high_pixel_high_conf'] = (conf_df['pixel_ratio'] >= HIGH_PIXEL_RATIO_THRESHOLD) & \
                                            (conf_df['avg_confidence'] >= HIGH_CONFIDENCE_THRESHOLD)
            
            combination_stats = conf_df.groupby('high_pixel_high_conf').agg({
                'count_accuracy': ['mean', 'std', 'count']
            }).round(4)
            combination_stats.columns = ['_'.join(col).strip() for col in combination_stats.columns]
            combination_stats = combination_stats.reset_index()
            
            categories = ['Other', 'High Pixel + High Conf']
            values = [
                combination_stats[~combination_stats['high_pixel_high_conf']]['count_accuracy_mean'].iloc[0] if any(~combination_stats['high_pixel_high_conf']) else 0,
                combination_stats[combination_stats['high_pixel_high_conf']]['count_accuracy_mean'].iloc[0] if any(combination_stats['high_pixel_high_conf']) else 0
            ]
            
            colors = ['red', 'green']
            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_ylabel('Mean Accuracy')
            ax4.set_title('Combined High Pixel Ratio + High Confidence')
            ax4.grid(True, alpha=0.3)
            
            # Add counts as text
            for bar, cat in zip(bars, categories):
                if cat == 'Other':
                    count = combination_stats[~combination_stats['high_pixel_high_conf']]['count_accuracy_count'].iloc[0] if any(~combination_stats['high_pixel_high_conf']) else 0
                else:
                    count = combination_stats[combination_stats['high_pixel_high_conf']]['count_accuracy_count'].iloc[0] if any(combination_stats['high_pixel_high_conf']) else 0
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'n={count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_5_pixel_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistical verification
        pixel_corr_significant = abs(corr) > 0.3 and p_val < 0.05
        
        if conf_analysis:
            conf_corr_significant = abs(conf_corr) > 0.3 and conf_p < 0.05
            
            # Test if high pixel + high confidence group performs better
            high_combo_data = conf_df[conf_df['high_pixel_high_conf']]['count_accuracy']
            other_data = conf_df[~conf_df['high_pixel_high_conf']]['count_accuracy']
            
            if len(high_combo_data) > 0 and len(other_data) > 0:
                combo_t, combo_p = stats.ttest_ind(high_combo_data, other_data, equal_var=False)
                combo_better = high_combo_data.mean() > other_data.mean() and combo_p < 0.05
            else:
                combo_better = False
        else:
            conf_corr_significant = False
            combo_better = False
        
        result = {
            'verified': pixel_corr_significant and (conf_corr_significant or combo_better),
            'pixel_correlation': {'r': corr, 'p': p_val, 'significant': pixel_corr_significant},
            'confidence_correlation': {'r': conf_corr, 'p': conf_p, 'significant': conf_corr_significant} if conf_analysis else None,
            'combined_effect': combo_better if conf_analysis else None,
            'conclusion': f"{'✅ VERIFIED' if pixel_corr_significant and (conf_corr_significant or combo_better) else '❌ PARTIALLY VERIFIED'}: "
                         f"Pixel ratio correlation: {pixel_corr_significant}, Combined effect: {combo_better if conf_analysis else 'No data'}"
        }
        
        self.hypothesis_results['H5_pixel_confidence'] = result
        print(f"📊 Pixel ratio correlation: r={corr:.3f}, p={p_val:.3f}")
        if conf_analysis:
            print(f"📊 Confidence correlation: r={conf_corr:.3f}, p={conf_p:.3f}")
            print(f"🎯 Combined high pixel+confidence effect: {combo_better}")
        print(f"✨ {result['conclusion']}")
        
        return result
    
    def generate_hypothesis_summary(self):
        """Generate comprehensive summary of all hypothesis testing results."""
        print("\n" + "="*80)
        print("📋 HYPOTHESIS TESTING SUMMARY")
        print("="*80)
        
        summary = {
            'total_hypotheses': len(self.hypothesis_results),
            'verified_count': sum(1 for r in self.hypothesis_results.values() if r.get('verified', False)),
            'results': self.hypothesis_results,
            'overall_conclusion': None
        }
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Hypothesis verification status
        hypothesis_names = ['H1: Count\nDegradation', 'H2: Numeral\nvs Text', 'H3: Background\nEffect', 
                          'H4: Model\nHierarchy', 'H5: Pixel+Conf\nRelation']
        verification_status = [
            self.hypothesis_results.get('H1_count_degradation', {}).get('verified', False),
            self.hypothesis_results.get('H2_numeral_vs_text', {}).get('verified', False),
            self.hypothesis_results.get('H3_background_effect', {}).get('verified', False),
            self.hypothesis_results.get('H4_model_hierarchy', {}).get('verified', False),
            self.hypothesis_results.get('H5_pixel_confidence', {}).get('verified', False)
        ]
        
        colors = ['green' if verified else 'red' for verified in verification_status]
        bars = ax1.bar(hypothesis_names, [1 if v else 0 for v in verification_status], color=colors, alpha=0.7)
        ax1.set_ylabel('Verification Status')
        ax1.set_title('Hypothesis Verification Results')
        ax1.set_ylim(0, 1.2)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add text labels
        for bar, verified in zip(bars, verification_status):
            text = '✅ VERIFIED' if verified else '❌ NOT VERIFIED'
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    text, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Model performance summary
        if 'H4_model_hierarchy' in self.hypothesis_results:
            hierarchy_data = self.hypothesis_results['H4_model_hierarchy']['overall_ranking']
            if hierarchy_data:
                models = [d['model_short'] for d in hierarchy_data]
                accuracies = [d['count_accuracy_mean'] for d in hierarchy_data]
                
                bars = ax2.bar(models, accuracies, color=[MODEL_COLORS.get(m, 'gray') for m in models])
                ax2.set_ylabel('Mean Accuracy')
                ax2.set_title('Overall Model Performance Ranking')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Critical insights summary
        critical_insights = [
            f"Count degradation: {'✅' if self.hypothesis_results.get('H1_count_degradation', {}).get('verified', False) else '❌'}",
            f"Text > Numeral: {'✅' if self.hypothesis_results.get('H2_numeral_vs_text', {}).get('verified', False) else '❌'}",
            f"Background hurts: {'✅' if self.hypothesis_results.get('H3_background_effect', {}).get('verified', False) else '❌'}",
            f"Model hierarchy: {'✅' if self.hypothesis_results.get('H4_model_hierarchy', {}).get('verified', False) else '❌'}",
            f"Pixel+Conf key: {'✅' if self.hypothesis_results.get('H5_pixel_confidence', {}).get('verified', False) else '❌'}"
        ]
        
        ax3.text(0.1, 0.9, '\n'.join(critical_insights), transform=ax3.transAxes, 
                fontsize=14, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
        ax3.set_title('Critical Insights Summary')
        ax3.axis('off')
        
        # Plot 4: Actionable recommendations
        recommendations = [
            "• Avoid high count prompts (≥7) for critical applications",
            "• Use word-based numbers over numerals when possible", 
            "• Minimize background descriptions in count prompts",
            "• Consider newer models but expect high-count failures",
            "• Monitor YOLO confidence + pixel ratios for quality"
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(recommendations), transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
        ax4.set_title('Actionable Recommendations')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text summary
        verified_count = sum(1 for r in self.hypothesis_results.values() if r.get('verified', False))
        total_count = len(self.hypothesis_results)
        
        summary['overall_conclusion'] = f"DMCAF Analysis: {verified_count}/{total_count} hypotheses verified"
        
        # Save detailed results
        with open(self.output_dir / 'hypothesis_analysis_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Overall Results: {verified_count}/{total_count} hypotheses verified")
        print(f"📁 Detailed results saved to: {self.output_dir / 'hypothesis_analysis_results.json'}")
        
        return summary
    
    def run_complete_analysis(self):
        """Run all hypothesis analyses in sequence."""
        print("🚀 Starting DMCAF Hypothesis-Driven Analysis")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run each hypothesis analysis
        self.analyze_hypothesis_1_count_degradation()
        self.analyze_hypothesis_2_numeral_vs_text()
        self.analyze_hypothesis_3_background_effect()
        self.analyze_hypothesis_4_model_hierarchy()
        self.analyze_hypothesis_5_pixel_confidence_relationship()
        
        # Generate summary
        summary = self.generate_hypothesis_summary()
        
        print("\n" + "="*80)
        print("🎉 DMCAF Hypothesis Analysis Complete!")
        print(f"📊 Results: {summary['verified_count']}/{summary['total_hypotheses']} hypotheses verified")
        print(f"📁 All plots and data saved to: {self.output_dir}")
        print("="*80)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="DMCAF Hypothesis-Driven Count Adherence Analysis")
    parser.add_argument("data_folder", help="Data folder containing metrics.db (e.g., 'data_003')")
    parser.add_argument("--output", default="hypothesis_analysis_results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Construct database path
    db_path = Path(args.data_folder) / "metrics.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    # Run analysis
    analyzer = HypothesisAnalyzer(str(db_path), args.output)
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()
