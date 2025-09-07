"""Run color analysis plots for DMCAF.

This module exposes a single entrypoint `run_color_analysis` that reads metrics
from the provided SQLite database and writes plots under the given output
directory (e.g., <datafolder>/analysis/color).
"""
import sqlite3
from pathlib import Path

from . import analyzer, plotter


def run_color_analysis(db_path: str, output_dir: str, verbose: bool = False) -> None:
    """Generate all color analysis plots.

    Args:
        db_path: Absolute path to metrics.db.
        output_dir: Directory where plots will be saved.
        verbose: If True, print intermediate summaries.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Quick schema check to ensure required columns exist
    try:
        conn = sqlite3.connect(db_path)
        pragma = conn.execute("PRAGMA table_info('image_evaluations')").fetchall()
        conn.close()
        cols = {row[1] for row in pragma}
        required = {"model_name", "color_accuracy"}
        if not required.issubset(cols):
            missing = ", ".join(sorted(required - cols))
            print(f"[Color] Skipping color analysis: missing required columns: {missing}")
            return
    except Exception as e:
        print(f"[Color] Skipping color analysis: unable to inspect DB: {e}")
        return

    # Average color accuracy by model
    try:
        accuracy_by_model = analyzer.calculate_average_color_accuracy_by_model(db_path, print_enabled=verbose)
        if accuracy_by_model:
            plotter.plot_accuracy_by_model(
                accuracy_by_model,
                save_path=str(out / "average_color_accuracy_by_model.png"),
            )
    except Exception as e:
        print(f"[Color] Skipped 'accuracy by model': {e}")

    # Average color accuracy by confidence bin
    try:
        accuracy_by_confidence_bin = analyzer.calculate_average_color_accuracy_by_confidence(db_path, print_enabled=verbose)
        if accuracy_by_confidence_bin:
            plotter.plot_accuracy_by_confidence_bin(
                accuracy_by_confidence_bin,
                save_path=str(out / "average_color_accuracy_by_confidence_bin.png"),
            )
    except Exception as e:
        print(f"[Color] Skipped 'accuracy by confidence bin': {e}")

    # Average color accuracy by object
    try:
        accuracy_by_object = analyzer.calculate_average_color_accuracy_by_object(db_path, print_enabled=verbose)
        if accuracy_by_object:
            plotter.plot_accuracy_by_object(
                accuracy_by_object,
                save_path=str(out / "average_color_accuracy_by_object.png"),
            )
    except Exception as e:
        print(f"[Color] Skipped 'accuracy by object': {e}")

    # Average color accuracy for low color variability objects
    try:
        accuracy_by_object_low_color_variability = analyzer.calculate_average_color_accuracy_for_low_color_variability_objects(db_path)
        if accuracy_by_object_low_color_variability:
            plotter.plot_accuracy_by_object_low_color_variability(
                accuracy_by_object_low_color_variability,
                save_path=str(out / "average_color_accuracy_for_low_color_variability.png"),
            )
    except Exception as e:
        print(f"[Color] Skipped 'low color variability objects': {e}")

    # Confusion matrix for misclassifications
    try:
        color_misclassifications = analyzer.calculate_count_by_detected_and_expected_color(db_path, print_enabled=verbose)
        if color_misclassifications:
            plotter.plot_confusion_matrix_detected_color_expected_color(
                color_misclassifications,
                save_path=str(out / "confusion_matrix_detected_vs_expected_color.png"),
            )
    except Exception as e:
        print(f"[Color] Skipped 'confusion matrix': {e}")

    # Average color accuracy by pixel coverage ratio
    try:
        coverages = analyzer.calculate_average_color_accuracy_by_pixel_ratio(db_path, print_enabled=verbose)
        if coverages:
            plotter.plot_average_color_accuracy_by_coverage(
                coverages,
                save_path=str(out / "average_color_accuracy_by_coverage.png"),
            )
    except Exception as e:
        print(f"[Color] Skipped 'accuracy by coverage': {e}")
