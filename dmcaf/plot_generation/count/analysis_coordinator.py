#!/usr/bin/env python3
"""
Coordinates the execution of all hypothesis analyses.
"""
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from .base_analyzer import BaseHypothesisAnalyzer, MODEL_COLORS
from .h1_count_degradation import Hypothesis1Analyzer
from .h2_numeral_vs_text import Hypothesis2Analyzer
from .h3_background_effect import Hypothesis3Analyzer
from .h4_model_hierarchy import Hypothesis4Analyzer
from .h5_pixel_confidence import Hypothesis5Analyzer
from .h6_suspicious_bbox import Hypothesis6Analyzer

warnings.filterwarnings('ignore')

class AnalysisCoordinator:
    """
    Coordinates the execution of all hypothesis analyses.
    """
    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Instantiate a base analyzer to load data once
        self.base_analyzer = BaseHypothesisAnalyzer(metrics_db_path, analytics_db_path, output_dir)
        self.base_analyzer.load_data()
        
        self.analyzers = [
            Hypothesis1Analyzer(metrics_db_path, analytics_db_path, output_dir),
            Hypothesis2Analyzer(metrics_db_path, analytics_db_path, output_dir),
            Hypothesis3Analyzer(metrics_db_path, analytics_db_path, output_dir),
            Hypothesis4Analyzer(metrics_db_path, analytics_db_path, output_dir),
            Hypothesis5Analyzer(metrics_db_path, analytics_db_path, output_dir),
            Hypothesis6Analyzer(metrics_db_path, analytics_db_path, output_dir),
        ]

        # Pass the loaded dataframe to each analyzer
        for analyzer in self.analyzers:
            analyzer.df = self.base_analyzer.df

        self.hypothesis_results = {}

    def run_complete_analysis(self):
        """Run all hypothesis analyses in sequence."""
        for analyzer in self.analyzers:
            result = analyzer.run_analysis()
            self.hypothesis_results[analyzer.hypothesis_name] = result
        
        summary = self.generate_hypothesis_summary()
        return summary

    def generate_hypothesis_summary(self):
        """Generate comprehensive summary of all hypothesis testing results."""
        summary = {
            'total_hypotheses': len(self.hypothesis_results),
            'verified_count': sum(1 for r in self.hypothesis_results.values() if r.get('verified', False)),
            'results': self.hypothesis_results,
            'overall_conclusion': None
        }
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Hypothesis verification status
        hypothesis_names = [
            'H1: Count\nDegradation', 'H2: Numeral\nvs Text', 'H3: Background\nEffect', 
            'H4: Model\nHierarchy', 'H5: Pixel+Conf\nRelation', 'H6: Suspicious\nBBox'
        ]
        verification_status = [
            self.hypothesis_results.get('H1_count_degradation', {}).get('verified', False),
            self.hypothesis_results.get('H2_numeral_vs_text', {}).get('verified', False),
            self.hypothesis_results.get('H3_background_effect', {}).get('verified', False),
            self.hypothesis_results.get('H4_model_hierarchy', {}).get('verified', False),
            self.hypothesis_results.get('H5_pixel_confidence_relationship', {}).get('verified', False),
            self.hypothesis_results.get('H6_suspicious_bbox_detection', {}).get('verified', False)
        ]
        
        colors = ['green' if verified else 'red' for verified in verification_status]
        bars = ax1.bar(hypothesis_names, [1 if v else 0 for v in verification_status], color=colors, alpha=0.7)
        ax1.set_ylabel('Verification Status')
        ax1.set_title('Hypothesis Verification Results')
        ax1.set_ylim(0, 1.2)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add text labels
        for bar, verified in zip(bars, verification_status):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, 
                     'VERIFIED' if verified else 'NOT VERIFIED', ha='center', va='bottom',
                     color=bar.get_facecolor(), fontweight='bold')
        
        # Plot 2: Model performance summary
        if 'H4_model_hierarchy' in self.hypothesis_results:
            ranking = self.hypothesis_results['H4_model_hierarchy'].get('overall_ranking', [])
            if ranking:
                ranking_df = pd.DataFrame(ranking)
                ranking_df.plot(kind='bar', x='model_short', y='count_accuracy_mean', ax=ax2,
                                yerr='count_accuracy_std', capsize=4, legend=False,
                                color=[MODEL_COLORS.get(m, 'gray') for m in ranking_df['model_short']])
                ax2.set_title('Overall Model Performance')
                ax2.set_ylabel('Mean Count Accuracy')
                ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Critical insights summary
        critical_insights = [
            f"Count degradation: {'✅' if self.hypothesis_results.get('H1_count_degradation', {}).get('verified', False) else '❌'}",
            f"Text > Numeral: {'✅' if self.hypothesis_results.get('H2_numeral_vs_text', {}).get('verified', False) else '❌'}",
            f"Background hurts: {'✅' if self.hypothesis_results.get('H3_background_effect', {}).get('verified', False) else '❌'}",
            f"Model hierarchy: {'✅' if self.hypothesis_results.get('H4_model_hierarchy', {}).get('verified', False) else '❌'}",
            f"Pixel+Conf key: {'✅' if self.hypothesis_results.get('H5_pixel_confidence_relationship', {}).get('verified', False) else '❌'}",
            f"Suspicious BBox: {'✅' if self.hypothesis_results.get('H6_suspicious_bbox_detection', {}).get('verified', False) else '❌'}"
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
            "• Monitor YOLO confidence + pixel ratios for quality",
            "• Use suspicious bbox detection to flag potential errors"
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
            json.dump(summary, f, indent=4, default=str)
        
        return summary
