#!/usr/bin/env python3
"""
Hypothesis 3: Background inclusion reduces count adherence due to attention split.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from .base_analyzer import BaseHypothesisAnalyzer

class Hypothesis3Analyzer(BaseHypothesisAnalyzer):
    """
    Analyzes the effect of including a background in the prompt on count adherence.
    """
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        super().__init__(metrics_db_path, analytics_db_path, output_dir)
        self.hypothesis_name = "H3_background_effect"
        self.hypothesis_text = "Background inclusion reduces count adherence due to attention split"

    def _cohens_d(self, x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    def run_analysis(self):
        """
        HYPOTHESIS 3: Background inclusion reduces count adherence due to attention split
        
        Method: Compare 'background' variant with 'base' variant with enhanced statistical visualization
        """
        # Filter for base and background variants only
        initial_variant_data = self.df[self.df['variant'].isin(['base', 'background'])].copy()
        
        if len(initial_variant_data) == 0:
            return {'verified': False, 'reason': 'No data available'}
        
        # Check if we need balanced sampling
        data_dist = initial_variant_data.groupby(['variant', 'model_short']).size().reset_index(name='count')
        expected_combinations = len(self.df['model_short'].unique()) * 2  # 2 variants
        
        if len(data_dist) < expected_combinations:
            variant_data = self._get_balanced_sample(initial_variant_data, ['base', 'background'])
        else:
            variant_data = initial_variant_data
        
        # Statistical comparison - enhanced visualization for presentation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Statistical significance testing with p-values and effect sizes
        model_comparisons = []
        for model in variant_data['model_short'].unique():
            model_df = variant_data[variant_data['model_short'] == model]
            
            base_acc = model_df[model_df['variant'] == 'base']['count_accuracy']
            bg_acc = model_df[model_df['variant'] == 'background']['count_accuracy']
            
            if len(base_acc) > 0 and len(bg_acc) > 0:
                t_stat, p_val = ttest_ind(base_acc, bg_acc, equal_var=False, nan_policy='omit')
                cohens_d = self._cohens_d(base_acc, bg_acc)
                model_comparisons.append({
                    'model': model,
                    'base_mean': base_acc.mean(),
                    'background_mean': bg_acc.mean(),
                    'difference': base_acc.mean() - bg_acc.mean(),
                    't_stat': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'base_better': base_acc.mean() > bg_acc.mean(),
                    'cohens_d': cohens_d
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
                p_marker = '***' if row.p_value < 0.001 else '**' if row.p_value < 0.01 else '*' if row.p_value < 0.05 else ''
                y_pos = bar.get_height()
                offset = 0.02 if y_pos >= 0 else -0.03
                ax1.text(bar.get_x() + bar.get_width() / 2, y_pos + offset, f"{p_marker}\nd={row.cohens_d:.2f}", 
                         ha='center', va='bottom' if y_pos >= 0 else 'top', fontsize=10, fontweight='bold')
            
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
                base_acc = variant_data[(variant_data['model_short'] == model) & (variant_data['variant'] == 'base')]['count_accuracy']
                bg_acc = variant_data[(variant_data['model_short'] == model) & (variant_data['variant'] == 'background')]['count_accuracy']
                
                if len(base_acc) > 1 and len(bg_acc) > 1:
                    diffs = [np.mean(np.random.choice(base_acc, len(base_acc))) - np.mean(np.random.choice(bg_acc, len(bg_acc))) for _ in range(1000)]
                    bootstrap_cis.append(np.percentile(diffs, [2.5, 97.5]))
                else:
                    bootstrap_cis.append([0, 0])
            
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
                y_pos = bar.get_height()
                offset = ci_upper_errors[i] + ci_range if y_pos >= 0 else -ci_lower_errors[i] - ci_range
                va = 'bottom' if y_pos >= 0 else 'top'
                ax2.text(bar.get_x() + bar.get_width() / 2, y_pos + offset, f"t={row.t_stat:.2f}\np={row.p_value:.3f}", 
                         ha='center', va=va, fontsize=9)
            
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
            ax3.plot(pivot_data.index, pivot_data['base'], marker='o', linewidth=3, label='Base', color='#2E8B57', markersize=8)
            ax3.plot(pivot_data.index, pivot_data['background'], marker='s', linewidth=3, label='Background', color='#DC143C', markersize=8)
            ax3.set_xlabel('Expected Count')
            ax3.set_ylabel('Mean Accuracy')
            ax3.set_title('Variant Performance by Expected Count')
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
        
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
            labels = ['Base', 'Background']
            coverage = [base_metrics['coverage_percentage'].iloc[0], bg_metrics['coverage_percentage'].iloc[0]]
            pixel_ratio = [base_metrics['pixel_ratio'].iloc[0], bg_metrics['pixel_ratio'].iloc[0]]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax4.bar(x - width/2, coverage, width, label='Coverage %', color='skyblue')
            ax4_twin = ax4.twinx()
            ax4_twin.bar(x + width/2, pixel_ratio, width, label='Pixel Ratio', color='salmon')
            
            ax4.set_ylabel('Coverage Percentage')
            ax4_twin.set_ylabel('Pixel Ratio')
            ax4.set_xticks(x)
            ax4.set_xticklabels(labels)
            ax4.set_title('Attention Split Analysis: Coverage vs Pixel Ratio')
            
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for coverage/pixel ratio analysis', 
                     ha='center', va='center', transform=ax4.transAxes)
        
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
        
        self.hypothesis_results[self.hypothesis_name] = result
        return result
