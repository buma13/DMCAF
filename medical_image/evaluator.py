import math, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
from monai.metrics import DiceMetric
from tqdm import tqdm

# ------------ CONFIG ------------
EXPERIMENT_NUMBER = 4
DATASET_TYPE      = "fundus"        # "fundus" or "decathlon"
THRESHOLD         = 0.8             # grayscale → binarise
# ---------------------------------

out_root   = Path.cwd() / "Outputs" / f"experiment{EXPERIMENT_NUMBER}"
images_dir = out_root / ("Segmentations" if DATASET_TYPE == "fundus" else "Images")
masks_dir  = out_root / "Masks"
RESULTS_FILE = out_root / "dice_scores.txt"

gen_files  = sorted(images_dir.glob("*.png"))
mask_files = sorted(masks_dir.glob("*.png"))
assert len(gen_files) == len(mask_files), "Image / mask count mismatch"

# -------- helpers --------
def load_png(p: Path) -> torch.Tensor:
    img = plt.imread(p)
    if img.ndim == 3:                     # RGB / RGBA
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.mean(-1)
    elif img.ndim == 1:                   # flattened
        side = int(math.isqrt(img.size))
        img = img.reshape(side, side)
    return torch.tensor(img, dtype=torch.float32)

def to_oh(x: torch.Tensor) -> torch.Tensor:
    """binary [H,W] → one-hot [1,2,H,W] float32"""
    return torch.stack([(1 - x), x], 0).unsqueeze(0).float()
# -------------------------

metric = DiceMetric(include_background=False, reduction="none")
scores, names = [], []

for g_fp, m_fp in tqdm(zip(gen_files, mask_files), total=len(gen_files), desc="Evaluating"):
    g = (load_png(g_fp) > THRESHOLD).to(torch.uint8)
    m = (load_png(m_fp) > THRESHOLD).to(torch.uint8)

    if DATASET_TYPE == "fundus":
        if m.sum() == 0:                 # empty GT → skip
            continue
        dice = metric(to_oh(g), to_oh(m)).item()
    else:                                # decathlon full image
        dice = metric(g.unsqueeze(0).unsqueeze(0).float(),
                      m.unsqueeze(0).unsqueeze(0).float()).item()

    scores.append(float(dice))
    names.append(g_fp.name)

# ---------- stats ----------
scores_arr       = np.array(scores)
nz_scores_arr    = scores_arr[scores_arr > 0]
nz_sample_count  = len(nz_scores_arr)

# ---------- save ----------
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    for n, s in zip(names, scores):
        f.write(f"{n}: Dice = {s:.4f}\n")

    f.write("\n==== All Samples ====\n")
    f.write(f"Samples Evaluated       : {len(scores)}\n")
    f.write(f"Mean Dice               : {scores_arr.mean():.4f}\n")
    f.write(f"Median Dice             : {np.median(scores_arr):.4f}\n")
    f.write(f"Std Dev Dice            : {scores_arr.std():.4f}\n")
    f.write(f"Min Dice                : {scores_arr.min():.4f}\n")
    f.write(f"Max Dice                : {scores_arr.max():.4f}\n")
    f.write(f"Dice > 0.90             : {(scores_arr > 0.9).mean()*100:.2f}%\n")
    f.write(f"Dice > 0.80             : {(scores_arr > 0.8).mean()*100:.2f}%\n")

    f.write("\n==== Non-zero Dice Only ====\n")
    f.write(f"Non-zero Samples        : {nz_sample_count}\n")
    if nz_sample_count:
        f.write(f"Mean Dice (nz)          : {nz_scores_arr.mean():.4f}\n")
        f.write(f"Median Dice (nz)        : {np.median(nz_scores_arr):.4f}\n")
        f.write(f"Std Dev Dice (nz)       : {nz_scores_arr.std():.4f}\n")
        f.write(f"Min Dice (nz)           : {nz_scores_arr.min():.4f}\n")
        f.write(f"Max Dice (nz)           : {nz_scores_arr.max():.4f}\n")
        f.write(f"Dice > 0.90 (nz)        : {(nz_scores_arr > 0.9).mean()*100:.2f}%\n")
        f.write(f"Dice > 0.80 (nz)        : {(nz_scores_arr > 0.8).mean()*100:.2f}%\n")

print("\nSaved summary to:", RESULTS_FILE)
