import argparse
import os
import yaml
from pathlib import Path

from dmcaf.plot_generation.count.analysis_coordinator import AnalysisCoordinator


def run_analysis(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Environment-provided base directory, consistent with other runners
    data_dir = os.getenv("OUTPUT_DIRECTORY")
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"[ERROR] data_dir does not exist: {data_dir}, Is the env variable \"OUTPUT_DIRECTORY\" set correctly?"
        )

    exp_id = config.get("experiment_id")
    if not exp_id:
        raise ValueError("experiment_id is required in analysis config")

    # DB paths
    metrics_db = os.path.join(data_dir, config.get("metrics_db", "metrics.db"))
    analytics_db = os.path.join(data_dir, config.get("analytics_db", "analytics.db"))

    # Output directory for plots under the same data dir, namespaced by experiment
    output_root = os.path.join(data_dir, "analysis")
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # Which analyses to run
    analyses = config.get("analyses", {"hypothesis_analysis": True})

    if analyses.get("hypothesis_analysis", True):
        print("[Analysis] Running hypothesis analysis...")
        coordinator = AnalysisCoordinator(
            metrics_db_path=metrics_db,
            analytics_db_path=analytics_db,
            output_dir=output_root,
        )
        summary = coordinator.run_complete_analysis()
        print("[Analysis] Hypothesis analysis complete.")
        print(f"Overall Conclusion: {summary['overall_conclusion']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMCAF analytics pipeline")
    parser.add_argument("config_path", type=str, help="Path to the analysis YAML config file")
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_analysis(args.config_path)
