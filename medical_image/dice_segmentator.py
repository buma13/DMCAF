import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from monai.metrics import DiceMetric
from tqdm import tqdm

# ---------- CONFIG ----------
GROUND_TRUTH_DIR = Path("Outputs/predictions/optic-disc")
PREDICTED_DIR    = Path("Outputs/predictions/Masks")
RESULTS_FILE     = Path("Outputs/predictions/dice_scores_foreground.txt")
N_SHOW           = 5          # how many pairs to visualise
# -----------------------------

# ---------- helpers ----------
def load_png(path: Path) -> torch.Tensor:
    """Load image, drop alpha, return grayscale float32 [H,W] in 0-1."""
    img = plt.imread(path)
    if img.ndim == 3:                      # RGB or RGBA
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.mean(-1)                # grayscale
    elif img.ndim == 1:                    # flattened row vector
        side = int(math.isqrt(img.size))
        if side * side != img.size:
            raise ValueError(f"{path.name}: cannot reshape {img.shape}")
        img = img.reshape(side, side)
    return torch.tensor(img, dtype=torch.float32)

def to_one_hot(bin_mask: torch.Tensor) -> torch.Tensor:
    """bin_mask [H,W]→ one-hot [1,2,H,W] (bg,fg) float32."""
    bg = (1 - bin_mask).unsqueeze(0)
    fg = bin_mask.unsqueeze(0)
    return torch.stack([bg, fg], dim=0).unsqueeze(0).float()
# -----------------------------

gt_files  = sorted(GROUND_TRUTH_DIR.glob("*.png"))
gen_files = sorted(PREDICTED_DIR.glob("*.png"))
assert len(gt_files) == len(gen_files), "File count mismatch."

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dice_metric = DiceMetric(include_background=False, reduction="none")

scores, names = [], []

# ---------- evaluation ----------
print(f"Total samples: {len(gt_files)}  |  Device: {device}")
for idx, (gt_fp, gen_fp) in enumerate(
        tqdm(zip(gt_files, gen_files), total=len(gt_files), desc="Evaluating")):

    gt_gray  = load_png(gt_fp)
    gen_gray = load_png(gen_fp)

    # non-black → 1
    gt_bin  = (gt_gray  > 0).to(torch.uint8)
    gen_bin = (gen_gray > 0).to(torch.uint8)

    # skip empty GT masks
    if gt_bin.sum() == 0:
        continue

    y      = to_one_hot(gt_bin).to(device)     # [1,2,H,W]
    y_pred = to_one_hot(gen_bin).to(device)

    dice = dice_metric(y_pred, y).item()
    scores.append(float(dice))
    names.append(gt_fp.name)

    # ---- optional visual check ----
    if idx < N_SHOW:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(gt_bin.cpu(),  cmap="gray", vmin=0, vmax=1)
        ax[0].set_title(f"GT bin  ({gt_fp.name})");  ax[0].axis("off")
        ax[1].imshow(gen_bin.cpu(), cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("Pred bin");                 ax[1].axis("off")
        plt.tight_layout(); plt.show()

# ---------- save results ----------
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    for n, s in zip(names, scores):
        f.write(f"{n}: Dice = {s:.4f}\n")

    f.write("\n==== Summary Statistics ====\n")
    f.write(f"Samples Evaluated       : {len(scores)}\n")
    f.write(f"Mean Dice               : {np.mean(scores):.4f}\n")
    f.write(f"Median Dice             : {np.median(scores):.4f}\n")
    f.write(f"Std Dev Dice            : {np.std(scores):.4f}\n")
    f.write(f"Min Dice                : {np.min(scores):.4f}\n")
    f.write(f"Max Dice                : {np.max(scores):.4f}\n")
    f.write(f"Dice > 0.90             : {(np.mean(np.array(scores) > 0.9)*100):.2f}%\n")
    f.write(f"Dice > 0.80             : {(np.mean(np.array(scores) > 0.8)*100):.2f}%\n")

print(f"\nSaved summary to: {RESULTS_FILE}")
