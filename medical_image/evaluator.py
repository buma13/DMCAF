import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm


#TODO Doesn't work with fundus dataset, only decathlon


# Define which experiment to evaluate
EXPERIMENT_NUMBER = 3  # Change this to the experiment number you want to evaluate

# Setup directories
output_dir = Path.cwd() / "Outputs"
experiment_dir = output_dir / f"experiment{EXPERIMENT_NUMBER}"
images_dir = experiment_dir / "Images"
masks_dir = experiment_dir / "Masks"
results_file = experiment_dir / "dice_scores.txt"

# Load image paths
generated_files = sorted(images_dir.glob("generated_*.png"))
mask_files = sorted(masks_dir.glob("mask_*.png"))
assert len(generated_files) == len(mask_files), "Mismatch between generated images and masks."

# Setup threshold and metric
threshold = 0.1 # from the original code, adjust as needed
to_discrete = AsDiscrete(threshold=threshold)
dice_metric = DiceMetric(include_background=False, reduction="none")

# Accumulate results
all_scores = []

for gen_file, mask_file in tqdm(zip(generated_files, mask_files), total=len(generated_files), desc="Evaluating"):
    gen_img = torch.tensor(plt.imread(gen_file), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask_img = torch.tensor(plt.imread(mask_file), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    pred_bin = to_discrete(gen_img)
    mask_bin = to_discrete(mask_img)

    dice = dice_metric(pred_bin, mask_bin)
    all_scores.append(dice.item())

# Save results
with open(results_file, "w") as f:
    for i, score in enumerate(all_scores):
        f.write(f"Sample {i}: Dice Score = {score:.4f}\n")
    f.write(f"\nAverage Dice Score: {np.mean(all_scores):.4f}\n")

print(f"Saved Dice scores to {results_file}")
