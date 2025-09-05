import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_by_model(accuracies_by_model, save_path):
    models = accuracies_by_model.keys()
    accuracies = accuracies_by_model.values()

    # Use a colormap for different bar colors
    colors = plt.cm.viridis([i / len(models) for i in range(len(models))])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(models, accuracies, color=colors)
    plt.ylabel('Average Color Accuracy')
    plt.xlabel('Model Name')
    plt.title('Average Color Accuracy by Model')
    # plt.ylim(0, 1)  # Y axis from 0 to 1
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_accuracy_by_confidence_bin(accuracies_by_confidence_bin, save_path):
    bins = accuracies_by_confidence_bin.keys()
    accuracies = accuracies_by_confidence_bin.values()

    # Use a different color for each bar
    bar_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']

    plt.figure(figsize=(8, 5))
    plt.bar(bins, accuracies, color=bar_colors)
    plt.xlabel('Confidence Bin')
    plt.ylabel('Average Color Accuracy')
    plt.title('Average Color Accuracy by Confidence Bin')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_accuracy_by_object(accuracies_by_object, save_path):
    objects = accuracies_by_object.keys()
    accuracies = accuracies_by_object.values()

    # Use a colormap for different bar colors
    colors = plt.cm.viridis([i / len(objects) for i in range(len(objects))])

    plt.figure(figsize=(10, 5))
    plt.bar(objects, accuracies, color=colors)
    plt.ylabel('Average Color Accuracy')
    plt.xlabel('Object')
    plt.title('Average Color Accuracy by Object')
    plt.xticks(rotation=90, ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_accuracy_by_object_low_color_variability(accuracies_by_object, save_path):
    objects = accuracies_by_object.keys()
    accuracies = accuracies_by_object.values()

    colors = ['navy']

    plt.figure(figsize=(10, 5))
    plt.bar(objects, accuracies, color=colors)
    plt.ylabel('Average Color Accuracy')
    plt.xlabel('Low Color Variability Object')
    plt.title('Average Color Accuracy for Low Color Variability Objects')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_confusion_matrix_detected_color_expected_color(misclassifications, save_path):
    detected_colors = set()
    expected_colors = set()
    pairs = []

    for m in misclassifications:
        detected = m['detected_color']
        expected = m['expected_color']
        count = m['count']
        detected_colors.add(detected)
        expected_colors.add(expected)
        pairs.append((detected, expected, int(count)))

    detected_colors = sorted(detected_colors)
    expected_colors = sorted(expected_colors)

    matrix = np.zeros((len(detected_colors), len(expected_colors)), dtype=int)

    for detected, expected, count in pairs:
        i = detected_colors.index(detected)
        j = expected_colors.index(expected)
        matrix[i, j] = count

    plt.figure(figsize=(12, 8))
    im = plt.imshow(matrix, cmap='Blues', aspect='auto')

    plt.xticks(np.arange(len(expected_colors)), expected_colors, rotation=45, ha='right')
    plt.yticks(np.arange(len(detected_colors)), detected_colors)

    # Write the count inside each cell
    for i in range(len(detected_colors)):
        for j in range(len(expected_colors)):
            if matrix[i, j] > 0:
                plt.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black', fontsize=8)

    plt.xlabel('Expected Color')
    plt.ylabel('Detected Color')
    plt.title('Confusion Matrix of Detected vs Expected Colors')
    plt.colorbar(im, label='Count')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_average_color_accuracy_by_coverage(coverages, save_path):
    plt.figure(figsize=(10, 6))
    for model, vals in coverages.items():
        plt.plot(vals['coverage'], vals['accuracy'], marker='o', label=model)

    plt.xlabel('Coverage (%)')
    plt.ylabel('Average Color Accuracy')
    plt.title('Average Color Accuracy by Coverage')
    plt.legend()
    plt.ylim(0.4, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()