# Adapted from MONAI
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import os
from pathlib import Path
import torch
import numpy as np
from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.utils import set_determinism
from models import SEED  # assuming models.py defines SEED and sets determinism

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

# Define transforms
train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        # transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(3.0, 3.0, 2.0), mode="bilinear"),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(64, 64, 44)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandSpatialCropd(keys=["image"], roi_size=(64, 64, 1), random_size=False),
        transforms.Lambdad(keys=["image"], func=lambda x: x.squeeze(-1)),
        transforms.CopyItemsd(keys=["image"], times=1, names=["mask"]),
        transforms.Lambdad(keys=["mask"], func=lambda x: torch.where(x > 0.1, 1, 0)),
        transforms.FillHolesd(keys=["mask"]),
        transforms.CastToTyped(keys=["mask"], dtype=np.float32),
    ]
)


"""
## Setup the brain tumour Decathlon dataset

We now download the Decathlon tumour dataset and extract the 2D slices from the 3D volumes. 
The images have four MRI channels: FLAIR, T1, T1c and T2. 
"""

def get_datasets_and_loaders(batch_size=8, num_workers=0, verbose=True):
    train_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=num_workers,
        download=True,
        seed=0,
        transform=train_transforms,
    )

    if verbose:
        print(f"Length of training data: {len(train_ds)}")
        print(f'Train image shape {train_ds[0]["image"].shape}')

    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="validation",
        cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
        num_workers=num_workers,
        download=True,
        seed=0,
        transform=train_transforms,
    )

    if verbose:
        print(f"Length of val data: {len(val_ds)}")
        print(f'Validation image shape {val_ds[0]["image"].shape}')

    persistent = True if num_workers >= 1 else False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, persistent_workers=persistent)

    return train_ds, val_ds, train_loader, val_loader
