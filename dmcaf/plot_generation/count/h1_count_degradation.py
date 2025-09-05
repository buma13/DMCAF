#!/usr/bin/env python3
"""
Hypothesis 1: Higher expected counts cause critical failure in all diffusion models.
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .base_analyzer import BaseHypothesisAnalyzer, CRITICAL_ACCURACY_THRESHOLD, MODEL_COLORS

class Hypothesis1Analyzer(BaseHypothesisAnalyzer):
    """
    Analyzes the degradation of count accuracy as expected counts increase.
    """
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        super().__init__(metrics_db_path, analytics_db_path, output_dir)
        self.hypothesis_name = "H1_count_degradation"
        self.hypothesis_text = "Higher expected counts cause critical failure in all diffusion models"

    def run_analysis(self):
        """
        HYPOTHESIS 1: Higher expected counts cause critical failure in all diffusion models
        
        Method: Analyze accuracy vs expected count patterns across all models
        """
        
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
        
        # Plot 2: Critical failure analysis (vectorized, pandas 2.2-safe)
        tmp = self.df.copy()
        tmp['is_critical'] = tmp['count_accuracy'] < CRITICAL_ACCURACY_THRESHOLD
        critical_failures = (
            tmp.groupby(['model_short', 'expected_count'])['is_critical']
               .mean()
               .reset_index(name='failure_rate')
        )
        
        for model in sorted(self.df['model_short'].unique()):
            model_data = critical_failures[critical_failures['model_short'] == model]
            ax2.plot(model_data['expected_count'], model_data['failure_rate'], 
                    marker='s', linewidth=2, label=model, color=MODEL_COLORS.get(model, 'black'))
        
        ax2.set_xlabel('Expected Count')
        ax2.set_ylabel('Critical Failure Rate')
        ax2.set_title('Critical Failure Rate by Expected Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Degradation correlation analysis (Spearman on per-count means)
        correlations = []
        for model in sorted(self.df['model_short'].unique()):
            # Use aggregated per-count mean accuracies to reduce noise and account for nonlinearity
            model_grp = grouped[grouped['model_short'] == model]
            if len(model_grp) >= 3:  # need at least 3 points for a meaningful rank correlation
                corr, p_val = spearmanr(model_grp['expected_count'], model_grp['count_accuracy_mean'])
                correlations.append({
                    'model': model,
                    'correlation': float(corr) if pd.notna(corr) else None,
                    'p_value': float(p_val) if pd.notna(p_val) else None,
                    'significant': (p_val is not None) and (p_val < 0.05) and (corr is not None) and (corr < 0),
                    'points': int(len(model_grp))
                })
        
        corr_df = pd.DataFrame(correlations)
        if not corr_df.empty:
            bars = ax3.bar(corr_df['model'], corr_df['correlation'], 
                           color=[MODEL_COLORS.get(m, 'gray') for m in corr_df['model']])
        else:
            bars = []
        
        # Color significant correlations differently
        for i, (bar, sig) in enumerate(zip(bars, corr_df['significant'])):
            if sig:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)

        ax3.set_ylabel('Spearman r (count vs mean accuracy)')
        ax3.set_title('Monotonic Degradation by Model (per-count means)')
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
                'critical_point': failure_rate >= 0.5
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
        # Consider a model as degrading if its Spearman correlation is negative and statistically significant (p<0.05)
        all_models_degrade = False
        if not corr_df.empty and 'significant' in corr_df.columns:
            all_models_degrade = bool(corr_df['significant'].all())
        
        # Calculate critical failure at high counts from threshold analysis
        high_count_critical_failure = any(threshold_df[threshold_df['expected_count'] >= 7]['critical_point']) if len(threshold_df) > 0 else False
        critical_failure_starts_at = min(critical_counts) if critical_counts else None
        
        result = {
            'verified': all_models_degrade and high_count_critical_failure,
            'correlations': corr_df.to_dict('records') if len(corr_df) > 0 else [],
            'threshold_analysis': threshold_df.to_dict('records') if len(threshold_df) > 0 else [],
            'critical_failure_starts_at': critical_failure_starts_at,
            'conclusion': f"{'✅ VERIFIED' if all_models_degrade and high_count_critical_failure else '❌ PARTIALLY VERIFIED'}: "
                         f"Monotonic degradation across models (Spearman), critical failure threshold: {critical_failure_starts_at}"
        }
        
        self.hypothesis_results[self.hypothesis_name] = result
        return result
