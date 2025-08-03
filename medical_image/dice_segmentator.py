import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from tqdm import tqdm

# ------------ CONFIG ------------
GROUND_TRUTH_DIR = Path("Outputs/predictions/optic-disc")
PREDICTED_DIR    = Path("Outputs/predictions/Masks")
RESULTS_FILE     = Path("Outputs/predictions") / "dice_scores.txt"
THRESHOLD        = 0.3
# --------------------------------

gt_files  = sorted(GROUND_TRUTH_DIR.glob("*.png"))
gen_files = sorted(PREDICTED_DIR.glob("*.png"))
assert len(gt_files) == len(gen_files), "Mismatch in file counts."

scores = []
file_names = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for gt_fp, gen_fp in tqdm(zip(gt_files, gen_files), total=len(gt_files), desc="Evaluating"):
    gt  = torch.tensor(plt.imread(gt_fp), dtype=torch.float32, device=device)
    gen = torch.tensor(plt.imread(gen_fp), dtype=torch.float32, device=device)

    if gt.ndim == 3:
        gt = gt.mean(-1)
    if gen.ndim == 3:
        gen = gen.mean(-1)

    pred_bin = (gen > THRESHOLD)
    mask_bin = (gt  > THRESHOLD)

    region = mask_bin
    if region.sum() == 0:
        continue

    inter = (pred_bin & mask_bin)[region].sum()
    union = pred_bin[region].sum() + mask_bin[region].sum()
    dice  = (2.0 * inter) / (union + 1e-8)

    scores.append(float(dice))
    file_names.append(gt_fp.name)

# ------------ SAVE ------------
with open(RESULTS_FILE, "w") as f:
    for fname, score in zip(file_names, scores):
        f.write(f"{fname}: Dice = {score:.4f}\n")

    f.write("\n==== Summary Statistics ====\n")
    f.write(f"Samples Evaluated       : {len(scores)}\n")
    f.write(f"Mean Dice               : {np.mean(scores):.4f}\n")
    f.write(f"Median Dice             : {np.median(scores):.4f}\n")
    f.write(f"Standard Deviation      : {np.std(scores):.4f}\n")
    f.write(f"Min Dice                : {np.min(scores):.4f}\n")
    f.write(f"Max Dice                : {np.max(scores):.4f}\n")
    f.write(f"Dice > 0.90 (success%)  : {(np.sum(np.array(scores) > 0.9) / len(scores)) * 100:.2f}%\n")
    f.write(f"Dice > 0.80             : {(np.sum(np.array(scores) > 0.8) / len(scores)) * 100:.2f}%\n")

print(f"Saved summary to: {RESULTS_FILE}")
