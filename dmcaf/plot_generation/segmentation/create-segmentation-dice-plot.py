import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """
    Generate average Dice score plot from segmentation_dice table.

    - Reads metrics.db from the folder set by OUTPUT_DIRECTORY env var.
    - Sorts classes by overall average Dice score (descending).
    - Saves the plot into a 'plots' folder next to this script.

    """
    output_dir_env = os.environ.get("OUTPUT_DIRECTORY")
    if not output_dir_env:
        raise EnvironmentError(
            "Environment variable OUTPUT_DIRECTORY is not set."
        )

    db_path = os.path.join(output_dir_env, "metrics.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        query = (
            "SELECT guidance_scale, class_name, dice_score "
            "FROM segmentation_dice"
        )
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError("No data found in segmentation_dice table")
    
    # Exclude "Cystic Duct"
    df = df[df["class_name"] != "Cystic Duct"]

    # Compute mean dice per (guidance_scale, class_name)
    grouped = (
        df.groupby(["guidance_scale", "class_name"])["dice_score"]
        .mean()
        .reset_index()
    )

    # Pivot for plotting: rows=class, cols=guidance_scale
    pivot = grouped.pivot(
        index="class_name",
        columns="guidance_scale",
        values="dice_score",
    )

    # Sort classes by overall (row-wise) mean dice score, descending
    class_order = pivot.mean(axis=1, skipna=True).sort_values(ascending=False).index
    pivot = pivot.loc[class_order]

    # Plot
    plt.figure(figsize=(9, 6))
    for guidance_scale in pivot.columns:
        plt.plot(
            pivot.index,
            pivot[guidance_scale],
            marker="o",
            label=f"{guidance_scale}",
        )
    
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Average Dice Score", fontsize=14)
    # plt.title("Average Dice Score by Class and Guidance Scale", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=14)
    plt.legend(title="Guidance Scale", fontsize=12, title_fontsize=12)
    # plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.grid()
    plt.tight_layout()

    # Ensure we always write to a plots/ folder next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "average_dice_score_over_classes.png")

    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
