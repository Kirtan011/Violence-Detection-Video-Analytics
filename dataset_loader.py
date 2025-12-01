"""
dataset_loader.py

Loads video clips (.npy) generated in Step 4 + fixed labels.csv.

Creates:
    * FightDataset class (PyTorch Dataset)
    * get_dataloaders() function → returns train_loader, val_loader

Expected structure:
    D:/violence-detection/workspace/clips/*.npy
    D:/violence-detection/workspace/clips/labels.csv
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


# ========= CONFIG ==========
ROOT = Path("D:/violence-detection").resolve()
CLIPS_DIR = ROOT / "workspace" / "clips"
LABELS_CSV = CLIPS_DIR / "labels.csv"

CLIP_LEN = 16
IMG_SIZE = 112

BATCH_SIZE = 16       # You can change this in training script
NUM_WORKERS = 4       # Adjust based on CPU cores
PIN_MEMORY = True
# ===========================


# ImageNet mean/std (same as training in Colab)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)


class FightDataset(Dataset):
    """
    Loads .npy clips of shape (16,112,112,3)
    and converts to torch tensor: (3,16,112,112)
    """
    def __init__(self, df):
        self.df = df
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        clip_path = self.paths[idx]
        label = self.labels[idx]

        arr = np.load(clip_path)  # shape (T,H,W,3)
        arr = arr.astype(np.float32) / 255.0

        # (T,H,W,3) -> (3,T,H,W)
        arr = np.transpose(arr, (3,0,1,2))
        tensor = torch.from_numpy(arr)

        # normalize
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

        return tensor, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=BATCH_SIZE, shuffle_train=True):
    """
    Reads labels.csv and returns:
        train_loader, val_loader
    """
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found at {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)

    # Add full path of .npy file
    df["path"] = df["clip_filename"].apply(lambda x: str(CLIPS_DIR / x))

    # Determine split from video_basename
    def get_split(vname):
        name = str(vname)
        if name.startswith("Train"):
            return "train"
        if name.startswith("Val"):
            return "val"
        return "unknown"

    df["split"] = df["video_basename"].apply(get_split)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    print("\nLoaded labels.csv:")
    print("Train clips:", len(train_df))
    print("Val clips:  ", len(val_df))

    train_dataset = FightDataset(train_df)
    val_dataset   = FightDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader


# Quick test if run directly
if __name__ == "__main__":
    print("Testing dataset loader...")
    train_loader, val_loader = get_dataloaders(batch_size=4)

    xb, yb = next(iter(train_loader))
    print("Batch shape:", xb.shape)     # (B,3,16,112,112)
    print("Labels:", yb)
