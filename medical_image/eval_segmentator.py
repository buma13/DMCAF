import os
from pathlib import Path
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

SEED = 42

# === Fundus dataset loader ===
def _build_fundus_lists(data_dir: Path, verbose: bool = True, test_size: float = 0.2):
    fundus_dir = data_dir / "Fundus"
    csv_path = fundus_dir / "metadata - standardized.csv"
    df = pd.read_csv(csv_path)
    df = df[df["fundus_od_seg"].notna()].reset_index(drop=True)

    def resolve(rel_path):
        return str(fundus_dir / rel_path.strip().lstrip("/"))

    img_paths = df["fundus"].apply(resolve)
    mask_paths = df["fundus_od_seg"].apply(resolve)

    items = [{"image": i, "mask": m} for i, m in zip(img_paths, mask_paths)
             if Path(i).is_file() and Path(m).is_file()]

    if verbose:
        print(f"Total with masks in CSV: {len(df)}")
        print(f"Valid image-mask pairs: {len(items)}")

    train_items, val_items = train_test_split(
        items, test_size=test_size, random_state=SEED, shuffle=True
    )

    return train_items, val_items

# === Segmentation Prediction ===
def run_fundus_segmentation(data_dir: Path):
    output_base = Path("Outputs") / "predictions"
    images_out = output_base / "Images"
    masks_out = output_base / "Masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    train_items, val_items = _build_fundus_lists(data_dir, verbose=True)
    all_items = train_items + val_items

    processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
    model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    for item in tqdm(all_items, desc="Segmenting fundus images"):
        img_path = Path(item["image"])
        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess and move to GPU
        inputs = processor(image_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # stays on GPU
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image_rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            pred_mask = upsampled_logits.argmax(dim=1)[0]  # still on GPU

        # Save image and mask (move to CPU only when saving)
        filename = img_path.name
        cv2.imwrite(str(images_out / filename), image_bgr)
        plt.imsave(masks_out / filename, pred_mask.cpu().numpy(), cmap="gray", vmin=0, vmax=2)

    print("Segmentation complete.")

# === Run ===
if __name__ == "__main__":
    dataset_root = Path("Datasets")  # Replace if needed
    run_fundus_segmentation(dataset_root)
