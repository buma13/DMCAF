import os
import analyzer
import plotter


def create_color_classification_analysis_plots():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "metrics.db")  # Update with the correct path if needed
    plot_path = os.path.join(current_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)

    accuracy_by_model = analyzer.calculate_average_color_accuracy_by_model(db_path)
    plotter.plot_accuracy_by_model(accuracy_by_model, save_path=os.path.join(plot_path, "average_color_accuracy_by_model.png"))

    accuracy_by_confidence_bin = analyzer.calculate_average_color_accuracy_by_confidence(db_path)
    plotter.plot_accuracy_by_confidence_bin(
        accuracy_by_confidence_bin, 
        save_path=os.path.join(plot_path, "average_color_accuracy_by_confidence_bin.png")
    )

    accuracy_by_object = analyzer.calculate_average_color_accuracy_by_object(db_path)
    plotter.plot_accuracy_by_object(
        accuracy_by_object, 
        save_path=os.path.join(plot_path, "average_color_accuracy_by_object.png")
    )

    accuracy_by_object_low_color_variability = analyzer.calculate_average_color_accuracy_for_low_color_variability_objects(db_path)
    plotter.plot_accuracy_by_object_low_color_variability(
        accuracy_by_object_low_color_variability, 
        save_path=os.path.join(plot_path, "average_color_accuracy_for_low_color_variability.png")
    )

    color_misclassifications = analyzer.calculate_count_by_detected_and_expected_color(db_path)
    plotter.plot_confusion_matrix_detected_color_expected_color(
        color_misclassifications, 
        save_path=os.path.join(plot_path, "confusion_matrix_detected_vs_expected_color.png")
    )

    coverages = analyzer.calculate_average_color_accuracy_by_pixel_ratio(db_path)
    plotter.plot_average_color_accuracy_by_coverage(
        coverages, 
        save_path=os.path.join(plot_path, "average_color_accuracy_by_coverage.png")
    )


if __name__ == "__main__":
    create_color_classification_analysis_plots()