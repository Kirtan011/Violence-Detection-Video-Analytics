"""
train_r3d18.py

Train an R3D-18 model for fight vs non-fight classification using
the clips generated in the preprocessing steps.

Expected structure:

    D:/violence-detection/
        workspace/
            clips/
                *.npy
                labels.csv
            models/
                (checkpoints will be saved here)

Uses:
    - dataset_loader.FightDataset via get_dataloaders()
    - R3D-18 (3D ResNet) from torchvision.models.video
"""

import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

import torchvision.models.video as video_models

from dataset_loader import get_dataloaders  # make sure PYTHONPATH includes training/ or run from root


# ========= CONFIG ==========
ROOT = Path("D:/violence-detection").resolve()
WORKSPACE = ROOT / "workspace"
MODELS_DIR = WORKSPACE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 2
BATCH_SIZE = 16       # must match (or override) dataset_loader default
NUM_EPOCHS = 15
BASE_LR = 5e-5
WEIGHT_DECAY = 1e-3
LABEL_SMOOTH = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================


def build_model(num_classes=2, device=DEVICE):
    """
    Build R3D-18 model and move to device.
    """
    print("Building R3D-18 model...")
    try:
        model = video_models.r3d_18(weights="KINETICS400_V1")
    except TypeError:
        # older torchvision versions
        model = video_models.r3d_18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


def accuracy(preds, labels):
    _, p = torch.max(preds, 1)
    return (p == labels).float().mean().item()


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, path):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
    torch.save(ckpt, path)


def load_checkpoint(model, optimizer, scheduler, path, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    epoch = ckpt.get("epoch", 0)
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    return epoch, best_val_acc


def main():
    print("Root:", ROOT)
    print("Workspace:", WORKSPACE)
    print("Models dir:", MODELS_DIR)
    print("Device:", DEVICE)

    # --------------------------
    # Data
    # --------------------------
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # --------------------------
    # Model, optimizer, scheduler
    # --------------------------
    model = build_model(NUM_CLASSES, DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",      # lower val_loss is better
        patience=2,
        factor=0.5,
        verbose=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # --------------------------
    # Resume if checkpoint exists
    # --------------------------
    last_ckpt_path = MODELS_DIR / "last_checkpoint.pth"
    best_model_path = MODELS_DIR / "best_r3d18_fight_classifier.pth"

    start_epoch = 0
    best_val_acc = 0.0

    if last_ckpt_path.exists():
        print(f"\n🔄 Found checkpoint at {last_ckpt_path}, loading...")
        start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, last_ckpt_path, device=DEVICE
        )
        start_epoch += 1  # resume from next epoch
        print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    else:
        print("\nNo checkpoint found, starting training from scratch.")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                out = model(xb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            train_loss += loss.item() * bs
            train_acc  += accuracy(out, yb) * bs
            n_train    += bs

            pbar.set_postfix(
                loss=f"{train_loss / max(1, n_train):.4f}",
                acc=f"{train_acc / max(1, n_train):.4f}",
            )

        train_loss /= max(1, n_train)
        train_acc  /= max(1, n_train)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
            for xb, yb in vbar:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    out = model(xb)
                    loss = criterion(out, yb)

                bs = xb.size(0)
                val_loss += loss.item() * bs
                val_acc  += accuracy(out, yb) * bs
                n_val    += bs

                vbar.set_postfix(
                    loss=f"{val_loss / max(1, n_val):.4f}",
                    acc=f"{val_acc / max(1, n_val):.4f}",
                )

        val_loss /= max(1, n_val)
        val_acc  /= max(1, n_val)

        # ---- Scheduler step ----
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # ---- Save last checkpoint ----
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_acc,
            last_ckpt_path,
        )
        print(f"💾 Saved last_checkpoint.pth at epoch {epoch+1}")

        # ---- Save best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 New BEST model saved! val_acc={best_val_acc:.4f}")

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)


if __name__ == "__main__":
    # If you run from project root, ensure Python sees training/ for dataset_loader:
    #   cd D:\violence-detection
    #   python -m training.train_r3d18
    #
    # Or add training/ to PYTHONPATH manually if running script directly.
    main()
