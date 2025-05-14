import os
import glob
import argparse
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, ScaleIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ToTensord, EnsureTyped
)

def get_brats2021_dataloaders(args):
    # Get all subject folders
    subject_dirs = glob.glob(os.path.join(args.data_dir, "BraTS2021_*"))
    
    # Create data dictionary
    data_dicts = []
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        data_dicts.append({
            "image": [
                os.path.join(subject_dir, f"{subject_id}_flair.nii.gz"),
                os.path.join(subject_dir, f"{subject_id}_t1.nii.gz"),
                os.path.join(subject_dir, f"{subject_id}_t1ce.nii.gz"),
                os.path.join(subject_dir, f"{subject_id}_t2.nii.gz")
            ],
            "label": os.path.join(subject_dir, f"{subject_id}_seg.nii.gz")
        })
    
    # Split into train and validation
    train_size = int(args.train_val_split * len(data_dicts))
    train_dicts = data_dicts[:train_size]
    val_dicts = data_dicts[train_size:]
    
    # Define transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=4
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ])
    
    # Create datasets
    train_ds = CacheDataset(
        data=train_dicts,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=args.num_workers
    )
    val_ds = CacheDataset(
        data=val_dicts,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=args.num_workers
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2021 Data Loader")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/BraTS2021_Training_Data",
                        help="Path to BraTS 2021 dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--train_val_split", type=float, default=0.8,
                        help="Fraction of data for training (rest for validation)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    args = parser.parse_args()
    
    train_loader, val_loader = get_brats2021_dataloaders(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")