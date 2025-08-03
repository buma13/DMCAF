import os
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assume you have these from previous imports
from models import get_model, get_controlnet, get_scheduler, get_inferers
from data import get_datasets_and_loaders

def random_resize_masks(masks, min_ratio=0.6, max_ratio=1.4, target_size=128):
    resized_masks = []
    for mask in masks:
        b, h, w = mask.shape[-3:]
        ratio = torch.empty(1).uniform_(min_ratio, max_ratio).item()
        new_size = int(h * ratio)

        # Resize mask
        resized = F.interpolate(mask.unsqueeze(0), size=(new_size, new_size), mode="nearest").squeeze(0)

        # Center-pad or crop to target size
        canvas = torch.zeros_like(mask)
        offset = (target_size - new_size) // 2

        if new_size <= target_size:
            canvas[:, offset:offset+new_size, offset:offset+new_size] = resized
        else:
            canvas = resized[:, -offset:target_size-offset, -offset:target_size-offset]

        resized_masks.append(canvas)
    return torch.stack(resized_masks)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base output directory
    current_dir = Path.cwd()
    output_base_dir = current_dir / "Outputs"

    # Create base if not exists
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Find next available experiment folder
    existing = [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith("experiment")]
    exp_nums = [int(d.name.replace("experiment", "")) for d in existing if d.name.replace("experiment", "").isdigit()]
    next_exp_num = max(exp_nums) + 1 if exp_nums else 0

    # Create new experiment directory
    output_dir = output_base_dir / f"experiment{next_exp_num}"
    masks_dir = output_dir / "Masks"
    images_dir = output_dir / "Images"

    masks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created experiment directory: {output_dir}")

    # Load models
    model = get_model(device, in_channels=1, out_channels=1)
    controlnet = get_controlnet(device, model, in_channels=1)
    scheduler = get_scheduler()
    _, controlnet_inferer = get_inferers(scheduler)

    # Load best checkpoints
    data_set = "fundus"  # Change to "fundus" if needed
    best_model_ckpt_path = current_dir / f"Checkpoints/DiffusionModelUNet/best_model_{data_set}.pth"
    best_controlnet_ckpt_path = current_dir / f"Checkpoints/ControlNet_DiffusionModelUNet/best_model_{data_set}.pth"
    fundus_dataset_dir= current_dir / "Datasets"
    model_ckpt = torch.load(best_model_ckpt_path, map_location=device)
    controlnet_ckpt = torch.load(best_controlnet_ckpt_path, map_location=device)

    model.load_state_dict(model_ckpt)
    controlnet.load_state_dict(controlnet_ckpt)

    model.eval()
    controlnet.eval()

    BATCH_SIZE = 16
    IMG_SIZE = 128
    NUM_WORKERS = 4
    DATASET = "fundus"  # Change to "fundus" if needed

    # Load val data
    _, val_ds, _, val_loader = get_datasets_and_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, verbose=False, dataset=DATASET, data_dir=fundus_dataset_dir)
    # Load val data
    for batch_idx, val_batch in enumerate(val_loader):
        val_images = val_batch["image"].to(device)
        val_masks = val_batch["mask"].to(device)
        cond_masks = random_resize_masks(val_masks, min_ratio=0.9, max_ratio=1.6, target_size=IMG_SIZE)

        num_samples = val_images.shape[0]
        sample = torch.randn((num_samples, 1, IMG_SIZE, IMG_SIZE)).to(device)

        for t in tqdm(scheduler.timesteps, desc=f"Sampling batch {batch_idx}", leave=False):
            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=sample, timesteps=torch.tensor([t]*num_samples).to(device).long(), controlnet_cond=val_masks
                )
                noise_pred = model(
                    sample,
                    timesteps=torch.tensor([t]*num_samples).to(device),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
                sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)

        for i in range(num_samples):
            mask_img = val_masks[i, 0].cpu().numpy()
            result_img = sample[i, 0].cpu().numpy()

            plt.imsave(masks_dir / f"mask_{batch_idx}_{i}.png", mask_img, cmap="gray", vmin=0, vmax=1)
            plt.imsave(images_dir / f"generated_{batch_idx}_{i}.png", result_img, cmap="gray", vmin=0, vmax=1)


if __name__ == "__main__":
    # Windows compatibility for multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()