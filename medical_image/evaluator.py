import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from tqdm import tqdm

# ------------ CONFIG ------------
EXPERIMENT_NUMBER = 3          # experimentX
DATASET_TYPE     = "fundus"    # "fundus" or "decathlon"
THRESHOLD        = 0.8         # brightness threshold
# --------------------------------

out_root       = Path.cwd() / "Outputs" / f"experiment{EXPERIMENT_NUMBER}"
images_dir     = out_root / ("Segmentations" if DATASET_TYPE == "fundus" else "Images")
masks_dir      = out_root / "Masks"
results_file   = out_root / "dice_scores.txt"

gen_files  = sorted(images_dir.glob("*.png"))
mask_files = sorted(masks_dir.glob("*.png"))
assert len(gen_files) == len(mask_files), "Image / mask count mismatch"

dice_metric = DiceMetric(include_background=False, reduction="none")
scores = []

for gen_fp, mask_fp in tqdm(zip(gen_files, mask_files),
                            total=len(gen_files), desc="Evaluating"):

    gen = torch.tensor(plt.imread(gen_fp),  dtype=torch.float32)
    msk = torch.tensor(plt.imread(mask_fp), dtype=torch.float32)

    # Ensure single-channel [H,W]
    if gen.ndim == 3:
        gen = gen.mean(-1)
    if msk.ndim == 3:
        msk = msk.mean(-1)

    if DATASET_TYPE == "fundus":
        pred_bin  = (gen > THRESHOLD)
        mask_bin  = (msk > THRESHOLD)

        region = mask_bin          # evaluate ONLY where GT is foreground
        if region.sum() == 0:      # no optic disc in this mask
            continue

        inter = (pred_bin & mask_bin)[region].sum()
        union = pred_bin[region].sum() + mask_bin[region].sum()
        dice  = (2.0 * inter) / (union + 1e-8)

    else:   # decathlon → full-image Dice
        pred_bin = (gen > THRESHOLD).unsqueeze(0).unsqueeze(0)
        mask_bin = (msk > THRESHOLD).unsqueeze(0).unsqueeze(0)
        dice     = dice_metric(pred_bin, mask_bin).item()

    scores.append(float(dice))

# ------------ SAVE ------------
with open(results_file, "w") as f:
    for i, s in enumerate(scores):
        f.write(f"Sample {i}: Dice = {s:.4f}\n")
    f.write(f"\nAverage Dice: {np.mean(scores):.4f}\n")

print("Saved to", results_file)
