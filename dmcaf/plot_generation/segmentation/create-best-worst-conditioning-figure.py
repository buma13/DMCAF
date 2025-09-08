import os
import sqlite3
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image


def _get_top_bottom_conditions(metrics_db: str, top_n: int = 2) -> List[Tuple[int, str, float]]:
    """Return top and bottom conditions by average dice score.

    Returns a list of tuples: (condition_id, image_path, avg_dice)
    ordered as top_n best followed by top_n worst.
    """
    conn = sqlite3.connect(metrics_db)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT condition_id, image_path, AVG(dice_score) AS avg_dice
        FROM segmentation_dice
        GROUP BY condition_id, image_path
        ORDER BY avg_dice DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    if len(rows) < top_n * 2:
        raise ValueError("Not enough data to select top and bottom conditions")
    best = rows[:top_n]
    worst = rows[-top_n:][::-1]  # lowest scores
    return best + worst


def _fetch_condition_data(conditioning_db: str, condition_id: int) -> Tuple[str, str, str]:
    """Fetch prompt, ground truth image path and segmentation path for a condition."""
    conn = sqlite3.connect(conditioning_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT prompt, image_path, segmentation_path FROM conditions WHERE id = ?",
        (condition_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Condition {condition_id} not found in conditioning DB")
    return row  # prompt, gt_image_path, gt_seg_path


def _load_image(path: str) -> Image.Image:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    base_dir = os.environ.get("OUTPUT_DIRECTORY")
    if not base_dir:
        raise EnvironmentError("Environment variable OUTPUT_DIRECTORY is not set.")

    conditioning_db = os.path.join(base_dir, "conditioning.db")
    metrics_db = os.path.join(base_dir, "metrics.db")

    rows = _get_top_bottom_conditions(metrics_db)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    column_titles = ["Prompt", "Ground Truth (GT)", "GT Segmentation", "Generated", "Generated Seg."]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title)

    for row_idx, (cond_id, gen_path, avg_dice) in enumerate(rows):
        prompt, gt_path, gt_seg_path = _fetch_condition_data(conditioning_db, cond_id)
        gen_seg_path = os.path.splitext(gen_path)[0] + "_segmented" + os.path.splitext(gen_path)[1]

        # Prompt text
        ax_prompt = axes[row_idx][0]
        ax_prompt.axis("off")
        ax_prompt.text(0.5, 0.5, prompt, ha="center", va="center", wrap=True)

        # Ground truth image
        axes[row_idx][1].imshow(_load_image(gt_path))
        axes[row_idx][1].axis("off")

        # GT segmentation
        axes[row_idx][2].imshow(_load_image(gt_seg_path))
        axes[row_idx][2].axis("off")

        # Generated image
        axes[row_idx][3].imshow(_load_image(gen_path))
        axes[row_idx][3].axis("off")

        # Generated segmentation
        axes[row_idx][4].imshow(_load_image(gen_seg_path))
        axes[row_idx][4].axis("off")

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "best_worst_conditionings.png")
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
