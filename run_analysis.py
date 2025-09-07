import argparse
import os
import yaml
from pathlib import Path

from dmcaf.plot_generation.count.count_analysis_runner import CountAnalysisRunner
from dmcaf.plot_generation.color.color_analysis_runner import run_color_analysis


def run_analysis_cli(experiment_id: str | None, run_count: bool, run_color: bool, config_path: str | None = None):
    """Unified entrypoint for analysis.

    - Preferred: pass experiment_id and select via --count/--color flags.
    - Optional: provide a legacy YAML config; we will read toggles but ignore per-hypothesis flags.
    """
    config = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Environment-provided base directory, consistent with other runners
    data_dir = os.getenv("OUTPUT_DIRECTORY")
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"[ERROR] data_dir does not exist: {data_dir}, Is the env variable \"OUTPUT_DIRECTORY\" set correctly?"
        )

    exp_id = experiment_id or config.get("experiment_id")
    if not exp_id:
        raise ValueError("experiment_id is required (pass --experiment-id or set in config)")

    # DB paths (config may override filenames)
    metrics_db = os.path.join(data_dir, config.get("metrics_db", "metrics.db"))
    analytics_db = os.path.join(data_dir, config.get("analytics_db", "analytics.db"))

    # Output directories for plots under the same data dir
    output_root = os.path.join(data_dir, "analysis")
    output_root_count = os.path.join(output_root, "count")
    output_root_color = os.path.join(output_root, "color")
    Path(output_root_count).mkdir(parents=True, exist_ok=True)
    Path(output_root_color).mkdir(parents=True, exist_ok=True)

    # If legacy config provided without flags, infer toggles
    if config_path and (not run_count and not run_color):
        analyses = config.get("analyses", {})
        run_count = any(
            analyses.get(k, False)
            for k in [
                "count_analysis",
                "numeral_vs_text",
                "background_effect",
                "model_hierarchy",
                "pixel_confidence",
                "suspicious_bboxes",
            ]
        )
        run_color = analyses.get("color_analysis", False)

    if run_count:
        print("[Analysis] Running count hypothesis analyses (H1–H6)...")
        runner = CountAnalysisRunner(
            metrics_db_path=metrics_db,
            analytics_db_path=analytics_db,
            output_dir=output_root_count,
        )
        summary = runner.run_complete_analysis()
        print("[Analysis] Count hypothesis analyses complete.")
        print(summary.get("overall_conclusion", ""))

    if run_color:
        print("[Analysis] Running color analyses...")
        run_color_analysis(db_path=metrics_db, output_dir=output_root_color)
        print("[Analysis] Color analyses complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMCAF analytics pipeline")
    parser.add_argument("--config", type=str, default=None, help="Optional path to analysis YAML config (legacy)")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID")
    parser.add_argument("--count", action="store_true", help="Run count hypothesis analyses (H1–H6)")
    parser.add_argument("--color", action="store_true", help="Run color analyses")
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_analysis_cli(args.experiment_id, args.count, args.color, args.config)
