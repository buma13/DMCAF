#!/usr/bin/env python3
"""
Hypothesis 2: Numeral prompts perform worse than base prompts (word-based numbers).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, wilcoxon
from .base_analyzer import BaseHypothesisAnalyzer, CRITICAL_ACCURACY_THRESHOLD

class Hypothesis2Analyzer(BaseHypothesisAnalyzer):
    """
    Analyzes the performance difference between numeral and text-based prompts.
    """
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        super().__init__(metrics_db_path, analytics_db_path, output_dir)
        self.hypothesis_name = "H2_numeral_vs_text"
        self.hypothesis_text = "Numeral prompts perform worse than base prompts (word-based numbers)"

    def _cohens_d(self, x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

    def run_analysis(self):
        """
        HYPOTHESIS 2: Numeral prompts perform worse than base prompts (word-based numbers)
        
        Method: Compare accuracy between 'numeral' and 'base' variants
        """
        # Filter for base and numeral variants only
        initial_variant_data = self.df[self.df['variant'].isin(['base', 'numeral'])].copy()
        
        if len(initial_variant_data) == 0:
            return {'verified': False, 'reason': 'No data available'}
        
        # Check if we need balanced sampling
        data_dist = initial_variant_data.groupby(['variant', 'model_short']).size().reset_index(name='count')
        expected_combinations = len(self.df['model_short'].unique()) * 2  # 2 variants
        
        if len(data_dist) < expected_combinations:
            variant_data = self._get_balanced_sample(initial_variant_data, ['base', 'numeral'])
        else:
            # Check if distribution is very uneven
            min_samples = data_dist['count'].min()
            max_samples = data_dist['count'].max()
            
            if max_samples > min_samples * 5:
                variant_data = self._get_balanced_sample(initial_variant_data, ['base', 'numeral'])
            else:
                variant_data = initial_variant_data
        
        # Statistical comparison - enhanced visualization for presentation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Statistical significance testing with p-values and effect sizes
        model_comparisons = []
        # Also collect monotonic (Spearman) and paired (Wilcoxon) stats per model
        for model in variant_data['model_short'].unique():
            model_df = variant_data[variant_data['model_short'] == model]
            
            base_acc = model_df[model_df['variant'] == 'base']['count_accuracy']
            numeral_acc = model_df[model_df['variant'] == 'numeral']['count_accuracy']
            
            if len(base_acc) > 0 and len(numeral_acc) > 0:
                t_stat, p_val = ttest_ind(base_acc, numeral_acc, equal_var=False, nan_policy='omit')
                cohens_d = self._cohens_d(base_acc, numeral_acc)

                # Per-count means for paired analysis (Wilcoxon test)
                per_count = (
                    model_df.groupby(['expected_count', 'variant'])['count_accuracy']
                    .mean()
                    .reset_index()
                    .pivot(index='expected_count', columns='variant', values='count_accuracy')
                )

                # Wilcoxon signed-rank on paired per-count differences (base - numeral)
                wilcoxon_stat = wilcoxon_p = None
                median_diff = None
                if 'base' in per_count.columns and 'numeral' in per_count.columns:
                    paired = per_count.dropna(subset=['base', 'numeral']).copy()
                    if len(paired) >= 3:
                        diffs = (paired['base'] - paired['numeral']).values
                        median_diff = float(np.median(diffs))
                        # Pratt handles zeros; alternative 'greater' tests base > numeral
                        try:
                            w_stat, w_p = wilcoxon(diffs, zero_method='pratt', alternative='greater')
                            wilcoxon_stat, wilcoxon_p = float(w_stat), float(w_p)
                        except ValueError:
                            # All diffs zero or insufficient variability
                            wilcoxon_stat, wilcoxon_p = None, None
                model_comparisons.append({
                    'model': model,
                    'base_mean': base_acc.mean(),
                    'numeral_mean': numeral_acc.mean(),
                    'difference': base_acc.mean() - numeral_acc.mean(),
                    't_stat': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'base_better': base_acc.mean() > numeral_acc.mean(),
                    'cohens_d': cohens_d,
                    # Wilcoxon paired test results
                    'wilcoxon_stat': wilcoxon_stat,
                    'wilcoxon_p': wilcoxon_p,
                    'median_diff': median_diff
                })
        
        comp_df = pd.DataFrame(model_comparisons)
        
        # Sort models in hierarchical order for consistent visualization
        def sort_models(df):
            if len(df) == 0:
                return df
            
            # Define the hierarchical order
            model_order = ['SD-v1-5', 'SD-2-1', 'SD-3-medium', 'SD-3.5-medium']
            
            # Create a categorical column for proper sorting
            df['model_cat'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
            df_sorted = df.sort_values('model_cat').drop('model_cat', axis=1).reset_index(drop=True)
            return df_sorted
        
        comp_df = sort_models(comp_df)
        
        # Enhanced statistical visualization with p-values and effect sizes
        if len(comp_df) > 0:
            # Create a more informative statistical plot
            x_pos = np.arange(len(comp_df))
            
            # Primary bars for accuracy differences
            colors = ['#2E8B57' if diff > 0 else '#DC143C' for diff in comp_df['difference']]
            bars = ax1.bar(x_pos, comp_df['difference'], color=colors, alpha=0.8, width=0.6)
            
            # Reserve headroom/footroom before adding text to prevent overflow
            y_min0, y_max0 = ax1.get_ylim()
            margin = (y_max0 - y_min0) * 0.25  # a bit more margin for multi-line labels
            ax1.set_ylim(y_min0 - margin, y_max0 + margin)
            
            # Add significance markers (Wilcoxon if available; fallback to t-test) and effect size annotations
            # Helper to keep text inside axes bounds
            y_min_lim, y_max_lim = ax1.get_ylim()
            y_span = (y_max_lim - y_min_lim) or 1.0
            pad = 0.03 * y_span
            for i, (bar, row) in enumerate(zip(bars, comp_df.itertuples())):
                p_used = row.wilcoxon_p if getattr(row, 'wilcoxon_p', None) is not None else row.p_value
                p_marker = '***' if p_used is not None and p_used < 0.001 else '**' if p_used is not None and p_used < 0.01 else '*' if p_used is not None and p_used < 0.05 else ''
                y_pos = bar.get_height()
                # Place inside the bar: below top for positive, above bottom for negative
                if y_pos >= 0:
                    y_anno = min(y_pos - pad, y_max_lim - pad)
                    va = 'top'
                else:
                    y_anno = max(y_pos + pad, y_min_lim + pad)
                    va = 'bottom'
                p_text = '<0.001' if (p_used is not None and p_used < 0.001) else (f"{p_used:.3f}" if p_used is not None else 'n/a')
                label_text = f"{p_marker}\np={p_text}\nd={row.cohens_d:.2f}"
                if getattr(row, 'median_diff', None) is not None:
                    label_text += f"\nΔ̃={row.median_diff:.2f}"
                ax1.text(bar.get_x() + bar.get_width() / 2, y_anno, label_text, 
                         ha='center', va=va, fontsize=9, fontweight='bold', clip_on=True)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([model.replace('SD-', '') for model in comp_df['model']], rotation=45)
            ax1.set_ylabel('Accuracy Difference (Base - Numeral)')
            ax1.set_title('Base vs Numeral: Wilcoxon (paired per-count) or t-test\n(*p<0.05, **p<0.01, ***p<0.001)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            
            # Add effect size interpretation legend (move to top-right, less clutter)
            ax1.text(0.98, 0.98, 'Effect Size (Cohen\'s d):\nSmall: 0.2, Medium: 0.5, Large: 0.8', 
                     transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            # Removed dense Spearman table to avoid clutter; Spearman stats will appear in Plot 2
        
        # Plot 2: Effect size analysis with interpretation
        if len(comp_df) > 0:
            # Create effect size interpretation plot
            effect_sizes = comp_df['cohens_d'].values
            models = [model.replace('SD-', '') for model in comp_df['model']]
            
            # Color code by effect size magnitude
            colors = []
            for d in effect_sizes:
                if abs(d) >= 0.8:
                    colors.append('darkred')
                elif abs(d) >= 0.5:
                    colors.append('red')
                elif abs(d) >= 0.2:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            bars = ax2.barh(models, effect_sizes, color=colors, alpha=0.8)
            
            # Add effect size magnitude labels
            for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
                label = 'Large' if abs(d) >= 0.8 else 'Medium' if abs(d) >= 0.5 else 'Small' if abs(d) >= 0.2 else 'Trivial'
                x_pos = d + (0.02 if d >= 0 else -0.02)
                ha = 'left' if d >= 0 else 'right'
                ax2.text(x_pos, i, label, va='center', ha=ha, fontsize=9, fontweight='bold')
            
            ax2.set_xlabel('Cohen\'s d (Effect Size)')
            ax2.set_title('Effect Size Analysis: Base vs Numeral\n(Positive = Base Better)')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            
            # Add reference lines for effect size thresholds
            ax2.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small (0.2)')
            ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Medium (0.5)')
            ax2.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large (0.8)')
            ax2.axvline(x=-0.2, color='orange', linestyle='--', alpha=0.5)
            ax2.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(x=-0.8, color='darkred', linestyle='--', alpha=0.5)
            
            # Add a subtle legend for thresholds
            ax2.text(0.98, 0.02, 'Effect Size Thresholds:\nSmall: ±0.2, Medium: ±0.5, Large: ±0.8', 
                    transform=ax2.transAxes, ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Percentage advantage/disadvantage by count
        count_variant_stats = variant_data.groupby(['expected_count', 'variant']).agg({
            'count_accuracy': 'mean'
        }).reset_index()
        
        pivot_data = count_variant_stats.pivot(index='expected_count', columns='variant', values='count_accuracy')
        
        if 'base' in pivot_data.columns and 'numeral' in pivot_data.columns:
            # Calculate percentage difference (Base - Numeral) * 100
            diff_pct = (pivot_data['base'] - pivot_data['numeral']) * 100
            
            # Color code: green for base advantage, red for numeral advantage
            colors = ['#2E8B57' if x > 0 else '#DC143C' for x in diff_pct]
            bars = ax3.bar(pivot_data.index, diff_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add zero reference line
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            
            # Add value labels on bars
            for bar, val in zip(bars, diff_pct):
                height = bar.get_height()
                # Position label above/below bar with some padding
                y_offset = 0.3 if height >= 0 else -0.8
                ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                        f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')
            
            ax3.set_xlabel('Expected Count')
            ax3.set_ylabel('Base Advantage (%)')
            ax3.set_title('Base vs Numeral: Percentage Advantage by Count\n(Positive = Base Better, Negative = Numeral Better)')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add full background shading to emphasize positive vs negative regions
            ax3.axhspan(0, ax3.get_ylim()[1], alpha=0.1, color='green', label='Base Advantage')
            ax3.axhspan(ax3.get_ylim()[0], 0, alpha=0.1, color='red', label='Numeral Advantage')
        
        # Plot 4: Simple direct comparison by count level with error bars
        if len(comp_df) > 0:
            # Show mean accuracy by count level with error bars
            count_stats = variant_data.groupby(['expected_count', 'variant']).agg({
                'count_accuracy': ['mean', 'std', 'count']
            }).reset_index()
            
            # Flatten column names
            count_stats.columns = ['expected_count', 'variant', 'mean_acc', 'std_acc', 'sample_size']
            
            base_data = count_stats[count_stats['variant'] == 'base']
            numeral_data = count_stats[count_stats['variant'] == 'numeral']
            
            # Calculate standard error for error bars
            base_data = base_data.copy()
            numeral_data = numeral_data.copy()
            base_data['se'] = base_data['std_acc'] / np.sqrt(base_data['sample_size'])
            numeral_data['se'] = numeral_data['std_acc'] / np.sqrt(numeral_data['sample_size'])
            
            # Plot with error bars
            ax4.errorbar(base_data['expected_count'], base_data['mean_acc'], 
                        yerr=base_data['se'], marker='o', linewidth=2, markersize=8,
                        label='Base (text)', color='#2E8B57', capsize=5, capthick=2)
            ax4.errorbar(numeral_data['expected_count'], numeral_data['mean_acc'], 
                        yerr=numeral_data['se'], marker='s', linewidth=2, markersize=8,
                        label='Numeral', color='#DC143C', capsize=5, capthick=2)
            
            # Mark significant differences at each count level
            y_max_data = max(base_data['mean_acc'].max(), numeral_data['mean_acc'].max())
            for count in sorted(base_data['expected_count'].unique()):
                base_acc = variant_data[(variant_data['variant'] == 'base') & 
                                       (variant_data['expected_count'] == count)]['count_accuracy']
                num_acc = variant_data[(variant_data['variant'] == 'numeral') & 
                                      (variant_data['expected_count'] == count)]['count_accuracy']
                
                if len(base_acc) > 0 and len(num_acc) > 0:
                    _, p_val = ttest_ind(base_acc, num_acc, equal_var=False)
                    if p_val < 0.05:
                        # Mark significant differences with a star
                        base_mean = base_acc.mean()
                        num_mean = num_acc.mean()
                        # Position asterisk well above the highest point with error bars
                        y_pos = y_max_data + 0.08  # Fixed position above data
                        color = 'green' if base_mean > num_mean else 'red'
                        ax4.text(count, y_pos, '*', ha='center', va='bottom', 
                                fontsize=20, fontweight='bold', color=color)
            
            
            ax4.set_xlabel('Expected Count')
            ax4.set_ylabel('Mean Accuracy ± SE')
            ax4.set_title('Base vs Numeral: Direct Comparison by Count\n(*significant difference, p<0.05)')
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            # Add critical threshold line
            ax4.axhline(y=CRITICAL_ACCURACY_THRESHOLD, color='red', linestyle=':', 
                       alpha=0.8, linewidth=2, label=f'Critical Threshold ({CRITICAL_ACCURACY_THRESHOLD})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_2_numeral_vs_text.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Conclusion
        if len(comp_df) > 0:
            # Use Wilcoxon decision per model when available; fallback to t-test
            model_supports = []
            for row in comp_df.itertuples():
                if getattr(row, 'wilcoxon_p', None) is not None and getattr(row, 'median_diff', None) is not None:
                    supports = (row.wilcoxon_p < 0.05) and (row.median_diff > 0)
                else:
                    supports = (row.p_value < 0.05) and (row.difference > 0)
                model_supports.append(bool(supports))
            base_better_count = int(sum(comp_df['base_better']))
            significant_base_better = int(sum(model_supports))
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
            'verification_method': 'Wilcoxon per-count paired (fallback t-test)',
            'conclusion': f"{'✅ VERIFIED' if hypothesis_verified else '❌ NOT VERIFIED'}: "
                         f"{significant_base_better}/{len(comp_df) if len(comp_df) > 0 else 0} models show significant base>numeral"
        }
        
        self.hypothesis_results[self.hypothesis_name] = result
        return result
