import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assume you have these from previous imports
from models import get_model, get_controlnet, get_scheduler, get_inferers
from data import get_datasets_and_loaders

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
model = get_model(device)
controlnet = get_controlnet(device, model, get_scheduler())
scheduler = get_scheduler()
_, controlnet_inferer = get_inferers(scheduler)

# Load best checkpoints
best_model_ckpt_path = current_dir / "Checkpoints/DiffusionModelUNet/best_model.pth"
best_controlnet_ckpt_path = current_dir / "Checkpoints/ControlNet_DiffusionModelUNet/best_model.pth"
model_ckpt = torch.load(best_model_ckpt_path, map_location=device)
controlnet_ckpt = torch.load(best_controlnet_ckpt_path, map_location=device)

model.load_state_dict(model_ckpt)
controlnet.load_state_dict(controlnet_ckpt)

model.eval()
controlnet.eval()

# Load val data
_, val_ds, _, val_loader = get_datasets_and_loaders(batch_size=8, num_workers=0, verbose=False)
# Load val data
for batch_idx, val_batch in enumerate(val_loader):
    val_images = val_batch["image"].to(device)
    val_masks = val_batch["mask"].to(device)

    num_samples = val_images.shape[0]
    sample = torch.randn((num_samples, 1, 64, 64)).to(device)

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

        plt.imsave(masks_dir / f"mask_{i}.png", mask_img, cmap="gray", vmin=0, vmax=1)
        plt.imsave(images_dir / f"generated_{i}.png", result_img, cmap="gray", vmin=0, vmax=1)
