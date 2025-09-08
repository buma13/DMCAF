#!/usr/bin/env python3
"""
Hypothesis 3: Background inclusion reduces count adherence due to attention split.
Align analysis methodology and visuals with H2 for consistency.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, wilcoxon
from .base_analyzer import BaseHypothesisAnalyzer, CRITICAL_ACCURACY_THRESHOLD

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
        
    Method: Compare 'background' vs 'base' variants using paired per-count Wilcoxon (fallback t-test),
    effect sizes, and count-wise comparisons, mirroring H2.
        """
        # Filter for base and background variants only
        initial_variant_data = self.df[self.df['variant'].isin(['base', 'background'])].copy()
        
        if len(initial_variant_data) == 0:
            return {'verified': False, 'reason': 'No data available'}
        
        # Check if we need balanced sampling (and guard against very uneven distributions)
        data_dist = initial_variant_data.groupby(['variant', 'model_short']).size().reset_index(name='count')
        expected_combinations = len(self.df['model_short'].unique()) * 2  # 2 variants

        if len(data_dist) < expected_combinations:
            variant_data = self._get_balanced_sample(initial_variant_data, ['base', 'background'])
        else:
            min_samples = data_dist['count'].min()
            max_samples = data_dist['count'].max()
            if max_samples > min_samples * 5:
                variant_data = self._get_balanced_sample(initial_variant_data, ['base', 'background'])
            else:
                variant_data = initial_variant_data
        
        # Statistical comparison - enhanced visualization for presentation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Statistical significance with per-model Wilcoxon (paired per-count), t-test fallback, and effect sizes
        model_comparisons = []
        for model in variant_data['model_short'].unique():
            model_df = variant_data[variant_data['model_short'] == model]

            base_acc = model_df[model_df['variant'] == 'base']['count_accuracy']
            bg_acc = model_df[model_df['variant'] == 'background']['count_accuracy']

            if len(base_acc) > 0 and len(bg_acc) > 0:
                t_stat, p_val = ttest_ind(base_acc, bg_acc, equal_var=False, nan_policy='omit')
                cohens_d = self._cohens_d(base_acc, bg_acc)

                # Per-count means for paired analysis (Wilcoxon test)
                per_count = (
                    model_df.groupby(['expected_count', 'variant'])['count_accuracy']
                    .mean()
                    .reset_index()
                    .pivot(index='expected_count', columns='variant', values='count_accuracy')
                )

                wilcoxon_stat = wilcoxon_p = None
                median_diff = None
                if 'base' in per_count.columns and 'background' in per_count.columns:
                    paired = per_count.dropna(subset=['base', 'background']).copy()
                    if len(paired) >= 3:
                        diffs = (paired['base'] - paired['background']).values
                        median_diff = float(np.median(diffs))
                        try:
                            w_stat, w_p = wilcoxon(diffs, zero_method='pratt', alternative='greater')
                            wilcoxon_stat, wilcoxon_p = float(w_stat), float(w_p)
                        except ValueError:
                            wilcoxon_stat, wilcoxon_p = None, None

                model_comparisons.append({
                    'model': model,
                    'base_mean': base_acc.mean(),
                    'background_mean': bg_acc.mean(),
                    'difference': base_acc.mean() - bg_acc.mean(),
                    't_stat': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'base_better': base_acc.mean() > bg_acc.mean(),
                    'cohens_d': cohens_d,
                    'wilcoxon_stat': wilcoxon_stat,
                    'wilcoxon_p': wilcoxon_p,
                    'median_diff': median_diff,
                })

        comp_df = pd.DataFrame(model_comparisons)

        # Sort models in hierarchical order for consistent visualization (same as H2)
        def sort_models(df):
            if len(df) == 0:
                return df
            model_order = ['SD-v1-5', 'SD-2-1', 'SD-3-medium', 'SD-3.5-medium']
            df['model_cat'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
            df_sorted = df.sort_values('model_cat').drop('model_cat', axis=1).reset_index(drop=True)
            return df_sorted

        comp_df = sort_models(comp_df)

        # Plot 1: difference bars with Wilcoxon/t-test annotations (mirroring H2)
        if len(comp_df) > 0:
            x_pos = np.arange(len(comp_df))
            colors = ['#2E8B57' if diff > 0 else '#DC143C' for diff in comp_df['difference']]
            bars = ax1.bar(x_pos, comp_df['difference'], color=colors, alpha=0.8, width=0.6)

            y_min0, y_max0 = ax1.get_ylim()
            margin = (y_max0 - y_min0) * 0.25
            ax1.set_ylim(y_min0 - margin, y_max0 + margin)

            y_min_lim, y_max_lim = ax1.get_ylim()
            y_span = (y_max_lim - y_min_lim) or 1.0
            pad = 0.03 * y_span
            for i, (bar, row) in enumerate(zip(bars, comp_df.itertuples())):
                p_used = row.wilcoxon_p if getattr(row, 'wilcoxon_p', None) is not None else row.p_value
                p_marker = '***' if p_used is not None and p_used < 0.001 else '**' if p_used is not None and p_used < 0.01 else '*' if p_used is not None and p_used < 0.05 else ''
                y_pos = bar.get_height()
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
            ax1.set_ylabel('Accuracy Difference (Base - Background)')
            ax1.set_title('Base vs Background: Wilcoxon (paired per-count)\n(*p<0.05, **p<0.01, ***p<0.001)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)

            ax1.text(0.98, 0.98, "Effect Size (Cohen's d):\nSmall: 0.2, Medium: 0.5, Large: 0.8",
                     transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Plot 2: Effect size analysis (same as H2 style)
        if len(comp_df) > 0:
            effect_sizes = comp_df['cohens_d'].values
            models = [model.replace('SD-', '') for model in comp_df['model']]
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
            for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
                label = 'Large' if abs(d) >= 0.8 else 'Medium' if abs(d) >= 0.5 else 'Small' if abs(d) >= 0.2 else 'Trivial'
                x_pos = d + (0.02 if d >= 0 else -0.02)
                ha = 'left' if d >= 0 else 'right'
                ax2.text(x_pos, i, label, va='center', ha=ha, fontsize=9, fontweight='bold')
            ax2.set_xlabel("Cohen's d (Effect Size)")
            ax2.set_title('Effect Size Analysis: Base vs Background\n(Positive = Base Better)')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            ax2.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small (0.2)')
            ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Medium (0.5)')
            ax2.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large (0.8)')
            ax2.axvline(x=-0.2, color='orange', linestyle='--', alpha=0.5)
            ax2.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(x=-0.8, color='darkred', linestyle='--', alpha=0.5)
            ax2.text(0.98, 0.02, 'Effect Size Thresholds:\nSmall: ±0.2, Medium: ±0.5, Large: ±0.8',
                     transform=ax2.transAxes, ha='right', va='bottom', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 3: Percentage advantage/disadvantage by count (Base - Background)
        count_variant_stats = variant_data.groupby(['expected_count', 'variant']).agg({
            'count_accuracy': 'mean'
        }).reset_index()
        pivot_data = count_variant_stats.pivot(index='expected_count', columns='variant', values='count_accuracy')
        if 'base' in pivot_data.columns and 'background' in pivot_data.columns:
            diff_pct = (pivot_data['base'] - pivot_data['background']) * 100
            colors = ['#2E8B57' if x > 0 else '#DC143C' for x in diff_pct]
            bars = ax3.bar(pivot_data.index, diff_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
            for bar, val in zip(bars, diff_pct):
                height = bar.get_height()
                y_offset = 0.3 if height >= 0 else -0.8
                ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                         f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=10, fontweight='bold')
            ax3.set_xlabel('Expected Count')
            ax3.set_ylabel('Base Advantage (%)')
            ax3.set_title('Base vs Background: Percentage Advantage by Count\n(Positive = Base Better, Negative = Background Better)')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.axhspan(0, ax3.get_ylim()[1], alpha=0.1, color='green', label='Base Advantage')
            ax3.axhspan(ax3.get_ylim()[0], 0, alpha=0.1, color='red', label='Background Advantage')

        # Plot 4: Direct comparison by count level with error bars and threshold (like H2)
        if len(comp_df) > 0:
            count_stats = variant_data.groupby(['expected_count', 'variant']).agg({
                'count_accuracy': ['mean', 'std', 'count']
            }).reset_index()
            count_stats.columns = ['expected_count', 'variant', 'mean_acc', 'std_acc', 'sample_size']
            base_data = count_stats[count_stats['variant'] == 'base'].copy()
            background_data = count_stats[count_stats['variant'] == 'background'].copy()
            base_data['se'] = base_data['std_acc'] / np.sqrt(base_data['sample_size'])
            background_data['se'] = background_data['std_acc'] / np.sqrt(background_data['sample_size'])
            ax4.errorbar(base_data['expected_count'], base_data['mean_acc'],
                         yerr=base_data['se'], marker='o', linewidth=2, markersize=8,
                         label='Base (text)', color='#2E8B57', capsize=5, capthick=2)
            ax4.errorbar(background_data['expected_count'], background_data['mean_acc'],
                         yerr=background_data['se'], marker='s', linewidth=2, markersize=8,
                         label='Background', color='#DC143C', capsize=5, capthick=2)
            y_max_data = max(base_data['mean_acc'].max(), background_data['mean_acc'].max())
            for count in sorted(base_data['expected_count'].unique()):
                base_acc = variant_data[(variant_data['variant'] == 'base') &
                                        (variant_data['expected_count'] == count)]['count_accuracy']
                bg_acc = variant_data[(variant_data['variant'] == 'background') &
                                      (variant_data['expected_count'] == count)]['count_accuracy']
                if len(base_acc) > 0 and len(bg_acc) > 0:
                    _, p_val = ttest_ind(base_acc, bg_acc, equal_var=False)
                    if p_val < 0.05:
                        base_mean = base_acc.mean()
                        bg_mean = bg_acc.mean()
                        y_pos = y_max_data + 0.08
                        color = 'green' if base_mean > bg_mean else 'red'
                        ax4.text(count, y_pos, '*', ha='center', va='bottom',
                                 fontsize=20, fontweight='bold', color=color)
            ax4.set_xlabel('Expected Count')
            ax4.set_ylabel('Mean Accuracy ± SE')
            ax4.set_title('Base vs Background: Direct Comparison by Count\n(*significant difference, p<0.05)')
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=CRITICAL_ACCURACY_THRESHOLD, color='red', linestyle=':',
                        alpha=0.8, linewidth=2, label=f'Critical Threshold ({CRITICAL_ACCURACY_THRESHOLD})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hypothesis_3_background_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Conclusion (use Wilcoxon decision per model when available; fallback to t-test), mirroring H2
        if len(comp_df) > 0:
            model_supports = []
            for row in comp_df.itertuples():
                if getattr(row, 'wilcoxon_p', None) is not None and getattr(row, 'median_diff', None) is not None:
                    supports = (row.wilcoxon_p < 0.05) and (row.median_diff > 0)
                else:
                    supports = (row.p_value < 0.05) and (row.difference > 0)
                model_supports.append(bool(supports))
            base_better_count = int(sum(comp_df['base_better']))
            significant_base_better = int(sum(model_supports))
            hypothesis_verified = significant_base_better > len(comp_df) * 0.5
        else:
            hypothesis_verified = False
            base_better_count = 0
            significant_base_better = 0

        # Build detailed per-plot outputs to match H2 schema
        plot1_data = comp_df.to_dict('records') if len(comp_df) > 0 else []
        plot2_effect_sizes = [
            {'model': row['model'], 'cohens_d': float(row['cohens_d'])}
            for _, row in comp_df.iterrows()
        ] if len(comp_df) > 0 else []

        count_variant_stats = (
            variant_data.groupby(['expected_count', 'variant'])
            .agg(mean_acc=('count_accuracy', 'mean'), sample_size=('count_accuracy', 'size'))
            .reset_index()
        )
        pivot_means = count_variant_stats.pivot(index='expected_count', columns='variant', values='mean_acc')
        pivot_counts = count_variant_stats.pivot(index='expected_count', columns='variant', values='sample_size')
        plot3_series = []
        if {'base', 'background'}.issubset(set(pivot_means.columns)):
            for expected_count in sorted(pivot_means.index):
                base_mean = float(pivot_means.loc[expected_count, 'base']) if pd.notna(pivot_means.loc[expected_count, 'base']) else None
                bg_mean = float(pivot_means.loc[expected_count, 'background']) if pd.notna(pivot_means.loc[expected_count, 'background']) else None
                base_adv_pct = float((base_mean - bg_mean) * 100) if (base_mean is not None and bg_mean is not None) else None
                base_n = int(pivot_counts.loc[expected_count, 'base']) if 'base' in pivot_counts.columns and pd.notna(pivot_counts.loc[expected_count, 'base']) else None
                bg_n = int(pivot_counts.loc[expected_count, 'background']) if 'background' in pivot_counts.columns and pd.notna(pivot_counts.loc[expected_count, 'background']) else None
                plot3_series.append({
                    'expected_count': int(expected_count),
                    'base_mean': base_mean,
                    'background_mean': bg_mean,
                    'base_advantage_pct': base_adv_pct,
                    'base_n': base_n,
                    'background_n': bg_n,
                })

        plot4_series = []
        if len(comp_df) > 0:
            count_stats = variant_data.groupby(['expected_count', 'variant']).agg({'count_accuracy': ['mean', 'std', 'count']}).reset_index()
            count_stats.columns = ['expected_count', 'variant', 'mean_acc', 'std_acc', 'sample_size']
            count_stats['se'] = count_stats['std_acc'] / np.sqrt(count_stats['sample_size'])
            for expected_count in sorted(count_stats['expected_count'].unique()):
                row_base = count_stats[(count_stats['expected_count'] == expected_count) & (count_stats['variant'] == 'base')]
                row_bg = count_stats[(count_stats['expected_count'] == expected_count) & (count_stats['variant'] == 'background')]
                base_acc = variant_data[(variant_data['variant'] == 'base') & (variant_data['expected_count'] == expected_count)]['count_accuracy']
                bg_acc = variant_data[(variant_data['variant'] == 'background') & (variant_data['expected_count'] == expected_count)]['count_accuracy']
                p_val = None
                if len(base_acc) > 0 and len(bg_acc) > 0:
                    _, p_val = ttest_ind(base_acc, bg_acc, equal_var=False)
                plot4_series.append({
                    'expected_count': int(expected_count),
                    'base': {
                        'mean': float(row_base['mean_acc'].iloc[0]) if not row_base.empty else None,
                        'std': float(row_base['std_acc'].iloc[0]) if not row_base.empty else None,
                        'n': int(row_base['sample_size'].iloc[0]) if not row_base.empty else None,
                        'se': float(row_base['se'].iloc[0]) if not row_base.empty else None,
                    },
                    'background': {
                        'mean': float(row_bg['mean_acc'].iloc[0]) if not row_bg.empty else None,
                        'std': float(row_bg['std_acc'].iloc[0]) if not row_bg.empty else None,
                        'n': int(row_bg['sample_size'].iloc[0]) if not row_bg.empty else None,
                        'se': float(row_bg['se'].iloc[0]) if not row_bg.empty else None,
                    },
                    'p_value': float(p_val) if p_val is not None else None,
                    'significant': (p_val is not None and p_val < 0.05)
                })

        result = {
            'verified': hypothesis_verified,
            'comparisons': comp_df.to_dict('records') if len(comp_df) > 0 else [],
            'base_better_count': base_better_count,
            'significant_base_better': significant_base_better,
            'total_models': len(comp_df) if len(comp_df) > 0 else 0,
            'verification_method': 'Wilcoxon per-count paired (fallback t-test)',
            'plots': {
                'plot_1_stats_by_model': plot1_data,
                'plot_2_effect_sizes': plot2_effect_sizes,
                'plot_3_base_advantage_by_count': plot3_series,
                'plot_4_direct_comparison_by_count': plot4_series,
            },
            'conclusion': f"{'✅ VERIFIED' if hypothesis_verified else '❌ NOT VERIFIED'}: "
                         f"{significant_base_better}/{len(comp_df) if len(comp_df) > 0 else 0} models show significant base>background"
        }
        
        self.hypothesis_results[self.hypothesis_name] = result
        return result
