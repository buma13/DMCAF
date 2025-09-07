#!/usr/bin/env python3
"""
Runs all count-based hypothesis analyses (H1–H6) with robust error handling.

This replaces the previous AnalysisCoordinator and always runs all hypotheses.
If required database columns are missing, the runner will skip analysis and
report the reason instead of failing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from .base_analyzer import BaseHypothesisAnalyzer
from .h1_count_degradation import Hypothesis1Analyzer
from .h2_numeral_vs_text import Hypothesis2Analyzer
from .h3_background_effect import Hypothesis3Analyzer
from .h4_model_hierarchy import Hypothesis4Analyzer
from .h5_pixel_confidence import Hypothesis5Analyzer
from .h6_suspicious_bbox import Hypothesis6Analyzer

warnings.filterwarnings("ignore")


class CountAnalysisRunner:
    """Coordinates and runs all count hypothesis analyses (H1–H6)."""

    def __init__(self, metrics_db_path: str, analytics_db_path: str, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load the data once using the base analyzer
        self.base_analyzer = BaseHypothesisAnalyzer(metrics_db_path, analytics_db_path, output_dir)
        self.base_analyzer.load_data()

        # Always run all H1–H6
        analyzer_classes = [
            Hypothesis1Analyzer,
            Hypothesis2Analyzer,
            Hypothesis3Analyzer,
            Hypothesis4Analyzer,
            Hypothesis5Analyzer,
            Hypothesis6Analyzer,
        ]

        self.analyzers = [cls(metrics_db_path, analytics_db_path, output_dir) for cls in analyzer_classes]
        for analyzer in self.analyzers:
            analyzer.df = self.base_analyzer.df

        self.hypothesis_results: dict[str, dict] = {}

    def run_complete_analysis(self) -> dict:
        """Run all hypothesis analyses in sequence with per-analyzer protection."""
        # If loading failed or minimal columns missing, skip gracefully
        if getattr(self.base_analyzer, "df", None) is None:
            reason = getattr(self.base_analyzer, "load_error", "Missing or invalid input data")
            summary = {
                "skipped": True,
                "reason": reason,
                "results": {},
                "overall_conclusion": f"DMCAF Count Analysis skipped: {reason}",
            }
            # Persist a lightweight summary file
            with open(self.output_dir / "hypothesis_analysis_results.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4)
            return summary

        # Execute each analyzer with isolation
        for analyzer in self.analyzers:
            try:
                result = analyzer.run_analysis()
            except Exception as e:
                # Do not fail the whole run – capture error and continue
                result = {
                    "verified": False,
                    "error": str(e),
                    "skipped": True,
                    "reason": "Analyzer error or missing assets (handled gracefully)",
                }
            self.hypothesis_results[analyzer.hypothesis_name] = result

        summary = self.generate_hypothesis_summary()
        return summary

    def generate_hypothesis_summary(self) -> dict:
        """Generate a JSON summary of all hypothesis results (no plots)."""
        summary = {
            "total_hypotheses": len(self.hypothesis_results),
            "verified_count": sum(1 for r in self.hypothesis_results.values() if r.get("verified", False)),
            "results": self.hypothesis_results,
            "overall_conclusion": None,
        }

        verified_count = sum(1 for r in self.hypothesis_results.values() if r.get("verified", False))
        total_count = len(self.hypothesis_results)
        summary["overall_conclusion"] = f"DMCAF Analysis: {verified_count}/{total_count} hypotheses verified"

        with open(self.output_dir / "hypothesis_analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, default=str)

        return summary
