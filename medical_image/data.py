# Adapted from MONAI
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

from models import SEED  # assuming models.py defines SEED and sets determinism

#TODO Brain part is unnecessary, can be removed

# Ensure determinism once per process
set_determinism(SEED)

# Setup dataset root directory
directory = os.environ.get("MONAI_DATA_DIRECTORY")
default_path = Path.cwd() / "Datasets"
root_dir = Path(directory) if directory else default_path
root_dir.mkdir(parents=True, exist_ok=True)

"""
## Specify transforms

We use the following transforms:
- `LoadImaged`: load the nifti image
- `EnsureChannelFirstd`: to bring the channel into the first dimension
- `Lambdad`: custom transform to select the modality (0: T1, 1: T2 etc.)
- `EnsureChannelFirstd`: do it again to create a channel (as one channel was selected previously, the channel dim is gone)
- `EnsureTyped`: we ensure that the image is indeed a tensor
- `Orientationd`: we reorient the images to make sure that they match RAS
- `Spacingd`: we bring the image resolution to 3, 3, 2
- `CenterSpatialCropd`: We select the central 64x64x44 crop
- `ScaleIntensityRangePercentilesd`: we normalise with percentiles
- `RandSpatialCropd`: we select an axial slice per volume
- `Lambdad`: we squeeze along the axial dimension
- `CopyItemsd`: we copy the image into a "mask" element
- `Lambdad`: we threshold the mask (which is the image) to turn it into a binary mask.
- `FillHolesd`: with this, we remove any potential artifact that happened during the thresholding
- `CastToTyped`: we cast to float.32

"""""""""

# Channel choice, assert valid
channel = 0
assert channel in [0, 1, 2, 3], "Choose a valid channel"

# --------------------------------------------------------------------------- #
#                                TRANSFORMS                                   #
# --------------------------------------------------------------------------- #

# Brain MRI (Decathlon)
def select_channel(x):
    return x[channel, :, :, :]

def squeeze_last_dim(x):
    return x.squeeze(-1)

def threshold_mask(x):
    return torch.where(x > 0.1, 1, 0)

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=select_channel),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Spacingd(keys=["image"], pixdim=(3.0, 3.0, 2.0), mode="bilinear"),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(64, 64, 44)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandSpatialCropd(keys=["image"], roi_size=(64, 64, 1), random_size=False),
        transforms.Lambdad(keys=["image"], func=squeeze_last_dim),
        transforms.CopyItemsd(keys=["image"], times=1, names=["mask"]),
        transforms.Lambdad(keys=["mask"], func=threshold_mask),
        transforms.FillHolesd(keys=["mask"]),
        transforms.CastToTyped(keys=["mask"], dtype=np.float32),
    ]
)


# Fundus + optic-disc masks

def rgb_to_grayscale(x):
    return x.mean(dim=0, keepdim=True)

fundus_tf = transforms.Compose([
    transforms.LoadImaged(keys=["image", "mask"]),
    transforms.EnsureChannelFirstd(keys=["image", "mask"]),
    transforms.Lambdad(keys="image", func=rgb_to_grayscale),
    transforms.ResizeD(keys=["image", "mask"], spatial_size=(128, 128), mode=("bilinear", "nearest")),
    transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
    transforms.CastToTyped(keys=["image", "mask"], dtype=np.float32),
])



# --------------------------------------------------------------------------- #
#                          FUNDUS DATA PREP FUNCTION                          #
# --------------------------------------------------------------------------- #
def _build_fundus_lists(data_dir: Path, verbose: bool = True, test_size: float = 0.2):
    fundus_dir = data_dir / "Fundus"
    csv_path = fundus_dir / "metadata - standardized.csv"
    df = pd.read_csv(csv_path)

    df = df[df["fundus_od_seg"].notna()].reset_index(drop=True)

    def resolve(rel_path):
        rel_path = rel_path.strip().lstrip("/")
        return str(fundus_dir / rel_path)

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


def get_datasets_and_loaders(batch_size: int = 8,
                             num_workers: int = 0,
                             verbose: bool = True,
                             dataset: str = "decathlon",
                             data_dir: Path = None):
    """
    Parameters:
        - dataset: "decathlon" or "fundus"
        - data_dir: optional path to override data_dir (Path or str)
    """
    # Determine root directory
    if data_dir is not None:
        data_dir = Path(data_dir)
    else:
        env_dir = os.environ.get("MONAI_DATA_DIRECTORY")
        default_dir = Path.cwd() / "Datasets"
        data_dir = Path(env_dir) if env_dir else default_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset.lower() == "decathlon":
        train_ds = DecathlonDataset(
            root_dir=data_dir,
            task="Task01_BrainTumour",
            section="training",
            cache_rate=1.0,
            num_workers=num_workers,
            download=True,
            seed=0,
            transform=train_transforms,
        )
        val_ds = DecathlonDataset(
            root_dir=data_dir,
            task="Task01_BrainTumour",
            section="validation",
            cache_rate=1.0,
            num_workers=num_workers,
            download=True,
            seed=0,
            transform=train_transforms,
        )
        if verbose:
            print(f"Decathlon training samples:  {len(train_ds)}")
            print(f"First image shape: {train_ds[0]['image'].shape}")

    elif dataset.lower() == "fundus":
        train_list, val_list = _build_fundus_lists(data_dir=data_dir, verbose=verbose)
        train_ds = Dataset(train_list, transform=fundus_tf)
        val_ds = Dataset(val_list, transform=fundus_tf)

    else:
        raise ValueError("dataset must be 'decathlon' or 'fundus'")

    persistent = num_workers > 0
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, persistent_workers=persistent)

    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True, persistent_workers=persistent)

    return train_ds, val_ds, train_loader, val_loader