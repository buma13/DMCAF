#!/usr/bin/env python3
"""
Hypothesis 4: Model hierarchy exists but all converge to low accuracy at high counts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from .base_analyzer import BaseHypothesisAnalyzer, CRITICAL_ACCURACY_THRESHOLD, MODEL_COLORS

class Hypothesis4Analyzer(BaseHypothesisAnalyzer):
    """
    Analyzes the performance hierarchy of different models and their convergence at high counts.
    """
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        super().__init__(metrics_db_path, analytics_db_path, output_dir)
        self.hypothesis_name = "H4_model_hierarchy"
        self.hypothesis_text = "Model hierarchy exists but all converge to low accuracy at high counts"

    def run_analysis(self):
        """
        HYPOTHESIS 4: Model hierarchy exists but all converge to low accuracy at high counts
        
        Method: Compare model performance overall and at high counts specifically
        """
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
            count_ranks = grouped[grouped['expected_count'] == count].sort_values('count_accuracy', ascending=False)
            count_ranks['rank'] = range(1, len(count_ranks) + 1)
            ranking_data.append(count_ranks)

        ranking_df = pd.DataFrame(pd.concat(ranking_data))

        # Create ranking stability plot
        for model in ranking_df['model_short'].unique():
            model_ranks = ranking_df[ranking_df['model_short'] == model]
            ax2.plot(model_ranks['expected_count'], model_ranks['rank'], marker='o', label=model, color=MODEL_COLORS.get(model, 'gray'))

        ax2.set_xlabel('Expected Count')
        ax2.set_ylabel('Model Rank (1 = Best)')
        ax2.set_title('Model Ranking Stability Across Counts\n(Shows when hierarchy collapses)')
        ax2.set_ylim(ax2.get_ylim()[::-1])  # Invert y-axis (rank 1 at top)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks(range(1, len(ranking_df['model_short'].unique()) + 1))

        # Calculate and annotate hierarchy stability
        stability_metrics = []
        for count in sorted(ranking_df['expected_count'].unique()):
            rank_variance = ranking_df[ranking_df['expected_count'] == count]['rank'].var()
            stability_metrics.append({'expected_count': count, 'rank_variance': rank_variance})

        # Find hierarchy collapse point (where rank variance becomes low or rankings shuffle significantly)
        stability_df = pd.DataFrame(stability_metrics)
        if len(stability_df) > 0:
            low_variance_threshold = 0.5  # Heuristic for low variance
            collapse_points = stability_df[stability_df['rank_variance'] < low_variance_threshold]
            if not collapse_points.empty:
                collapse_count = collapse_points['expected_count'].min()
                ax2.axvline(x=collapse_count, color='purple', linestyle='--', alpha=0.8, label=f'Hierarchy Collapse (Count {collapse_count})')

        # Update legend to include new elements
        ax2.legend(fontsize=10, loc='lower right')
        
        # Plot 3: Model performance across count ranges
        count_bins = [(1, 3), (4, 6), (7, 10)]
        bin_labels = ['Low (1-3)', 'Medium (4-6)', 'High (7-10)']
        
        model_by_bins = {}
        for model in self.df['model_short'].unique():
            accuracies = []
            for b in count_bins:
                bin_df = self.df[(self.df['model_short'] == model) & (self.df['expected_count'] >= b[0]) & (self.df['expected_count'] <= b[1])]
                accuracies.append(bin_df['count_accuracy'].mean())
            model_by_bins[model] = accuracies
        
        x = np.arange(len(bin_labels))
        width = 0.2
        
        for i, model in enumerate(sorted(model_by_bins.keys())):
            offset = width * (i - 1.5)
            ax3.bar(x + offset, model_by_bins[model], width, label=model, color=MODEL_COLORS.get(model, 'gray'))
        
        ax3.set_xlabel('Count Range')
        ax3.set_ylabel('Mean Accuracy')
        ax3.set_title('Model Performance by Count Range')
        ax3.set_xticks(x)
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
                if len(model1_acc) > 1 and len(model2_acc) > 1:
                    t_stat, p_val = ttest_ind(model1_acc, model2_acc, equal_var=False, nan_policy='omit')
                    model_pairs.append({'model1': models[i], 'model2': models[j], 'p_value': p_val, 'significant': p_val < 0.05})
        
        pairs_df = pd.DataFrame(model_pairs)
        
        # Check convergence at high counts and ranking stability
        all_low_at_high = (model_high_count['count_accuracy_mean'] < CRITICAL_ACCURACY_THRESHOLD).all()
        hierarchy_exists = len(pairs_df[pairs_df['significant']]) > 0
        
        # Add ranking stability analysis to results
        hierarchy_collapse_point = None
        collapse_points = []
        
        # Check if stability analysis was performed in Plot 2
        if 'stability_df' in locals() and len(stability_df) > 0:
            low_variance_threshold = 0.5
            collapse_points_df = stability_df[stability_df['rank_variance'] < low_variance_threshold]
            if not collapse_points_df.empty:
                hierarchy_collapse_point = collapse_points_df['expected_count'].min()
        
        # Build per-plot numeric outputs
        # Plot 1: overall ranking stats already in model_overall
        plot1_data = model_overall.to_dict('records')

        # Plot 2: ranking stability per model time series
        plot2_series = {}
        for model in ranking_df['model_short'].unique():
            md = ranking_df[ranking_df['model_short'] == model].sort_values('expected_count')
            plot2_series[model] = [
                {
                    'expected_count': int(row['expected_count']),
                    'rank': int(row['rank'])
                }
                for _, row in md.iterrows()
            ]

        # Plot 3: performance across count bins
        plot3_bins = {
            model: [
                {
                    'bin_label': bin_labels[idx],
                    'mean_accuracy': float(acc) if pd.notna(acc) else None
                }
                for idx, acc in enumerate(vals)
            ]
            for model, vals in model_by_bins.items()
        }

        # Plot 4: high count variance and stats
        plot4_high = high_count_variance.copy()
        plot4_high = plot4_high.merge(model_high_count[['model_short','count_accuracy_count']], on='model_short', how='left')
        plot4_data = [
            {
                'model_short': row['model_short'],
                'mean': float(row['mean']),
                'std': float(row['std']),
                'n': int(row['count_accuracy_count']) if not pd.isna(row['count_accuracy_count']) else None
            }
            for _, row in plot4_high.iterrows()
        ]

        result = {
            'verified': hierarchy_exists and all_low_at_high,
            'overall_ranking': model_overall.to_dict('records'),
            'high_count_performance': model_high_count.to_dict('records'),
            'ranking_stability': stability_df.to_dict('records') if len(stability_df) > 0 else [],
            'hierarchy_collapse_point': hierarchy_collapse_point,
            'significant_pairs': len(pairs_df[pairs_df['significant']]),
            'total_pairs': len(pairs_df),
            'all_critical_at_high': all_low_at_high,
            'plots': {
                'plot_1_overall_ranking': plot1_data,
                'plot_2_ranking_stability': {
                    'series': plot2_series,
                    'collapse_threshold_variance': 0.5,
                },
                'plot_3_bins_performance': plot3_bins,
                'plot_4_high_count_variance': plot4_data,
            },
            'conclusion': f"{'✅ VERIFIED' if hierarchy_exists and all_low_at_high else '❌ PARTIALLY VERIFIED'}: "
                         f"Hierarchy exists: {hierarchy_exists}, Collapses at count: {hierarchy_collapse_point}, All critical at high counts: {all_low_at_high}"
        }
        
        self.hypothesis_results[self.hypothesis_name] = result
        return result
