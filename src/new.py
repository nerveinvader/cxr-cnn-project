"""
train_chestxray14.py – PyTorch 2.3 multi-label trainer for NIH ChestX-ray14.

Usage examples
--------------
# Full training run (GPU auto-detected)
python train_chestxray14.py --data_dir /data/ChestXray14 --epochs 20 --batch_size 32

# Quick CPU test
python train_chestxray14.py --data_dir ./ChestXray14 --epochs 2 --batch_size 8 --num_workers 0

# Inference on a single image
python train_chestxray14.py --predict ./test.png --checkpoint best_model.pth --data_dir ./ChestXray14
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.classification import MultilabelAUROC
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
FINDINGS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ChestXrayDataset(data.Dataset):
    """Custom Dataset for NIH ChestX-ray14 (PA view, multi-label)."""

    def __init__(
        self,
        csv_df: pd.DataFrame,
        img_dir: Path,
        transforms: A.Compose,
    ) -> None:
        self.df = csv_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

        # Pre-encode labels for speed (N, 14) float32
        self.targets = np.zeros((len(self.df), len(FINDINGS)), dtype=np.float32)
        for idx, labels in enumerate(self.df["Finding Labels"].values):
            if labels == "No Finding":
                continue
            for lab in labels.split("|"):
                self.targets[idx, FINDINGS.index(lab)] = 1.0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["Image Index"]
        image = A.imread(img_path.as_posix())  # BGR uint8
        augmented = self.transforms(image=image)
        img_tensor = augmented["image"]  # C×H×W float32
        target = torch.from_numpy(self.targets[idx])
        return img_tensor, target


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def build_transforms(train: bool) -> A.Compose:
    """Albumentations pipeline (320×320, CLAHE, flips, rot, normalize)."""
    base = [
        A.Resize(320, 320, interpolation=1),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True),
    ]
    if train:
        base += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=7, p=0.4),
        ]
    base += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(base)


def make_splits(
    csv_file: Path,
    seed: int = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Patient-level train/val/test splits."""
    df = pd.read_csv(csv_file)
    df = df[df["View Position"] == "PA"].copy()  # keep only PA
    patients = df["Patient ID"].unique()
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    test_patients = set(patients[: int(n * test_ratio)])
    val_patients = set(patients[int(n * test_ratio) : int(n * (test_ratio + val_ratio))])

    is_test = df["Patient ID"].isin(test_patients)
    is_val = df["Patient ID"].isin(val_patients)
    test_df = df[is_test]
    val_df = df[is_val]
    train_df = df[~(is_test | is_val)]
    return train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model() -> nn.Module:
    """EfficientNet-B0 backbone with 14-sigmoid head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_feat = model.classifier[1].in_features  # type: ignore[arg-type]
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_feat, len(FINDINGS))
    )
    return model


# --------------------------------------------------------------------------- #
# Training / Evaluation loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    loss_fn: nn.Module,
    device: torch.device,
    accum_steps: int = 1,
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Train", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = loss_fn(outputs, targets) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{running_loss/ (step+1):.4f}")

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    metric: MultilabelAUROC,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, float]:
    model.eval()
    metric.reset()
    running_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        running_loss += loss.item()
        preds = torch.sigmoid(outputs)
        metric.update(preds, targets.int())
    per_class = metric.compute().cpu().numpy()  # shape (14,)
    mean_auc = per_class.mean()
    return running_loss / len(loader), per_class, mean_auc


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_checkpoint(
    state: dict, is_best: bool, out_dir: Path, fname: str = "last.pth"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / fname)
    if is_best:
        torch.save(state, out_dir / "best_model.pth")


def predict_image(
    img_path: Path,
    checkpoint: Path,
    device: torch.device,
) -> dict[str, float]:
    """Load model & output 14 probabilities for a single image."""
    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    tfm = build_transforms(train=False)
    img = A.imread(img_path.as_posix())
    tensor = tfm(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        probs = torch.sigmoid(model(tensor)).cpu().squeeze().numpy()
    return {f: float(p) for f, p in zip(FINDINGS, probs)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA_DIR", "."))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", type=str, help="image path for inference")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"
    csv_file = data_dir / "Data_Entry_2017.csv"
    output_dir = Path(args.output_dir)
    ckpt_path = output_dir / args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    seed_everything(args.seed)

    # --------------------------------------------------------------------- #
    # Optional inference-only mode
    # --------------------------------------------------------------------- #
    if args.predict:
        result = predict_image(Path(args.predict), ckpt_path, device)
        print("Predicted probabilities:")
        for k, v in result.items():
            print(f"{k:>20s}: {v:.4f}")
        return

    # --------------------------------------------------------------------- #
    # Data loaders
    # --------------------------------------------------------------------- #
    train_df, val_df, test_df = make_splits(csv_file, seed=args.seed)
    train_ds = ChestXrayDataset(train_df, img_dir, build_transforms(train=True))
    val_ds = ChestXrayDataset(val_df, img_dir, build_transforms(train=False))
    test_ds = ChestXrayDataset(test_df, img_dir, build_transforms(train=False))

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------------------- #
    # Model, loss, optimizer, scheduler
    # --------------------------------------------------------------------- #
    model = build_model().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metric = MultilabelAUROC(num_labels=len(FINDINGS), average=None).to(device)

    # --------------------------------------------------------------------- #
    # Training loop with early stopping
    # --------------------------------------------------------------------- #
    best_auc = 0.0
    epochs_no_improve = 0
    patience = 5
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}] – LR {scheduler.get_last_lr()[0]:.2e}")

        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_fn,
            device,
            accum_steps=args.accum_steps,
        )
        scheduler.step(epoch + len(train_loader) / len(train_loader))  # warm restart step

        val_loss, per_class_auc, mean_auc = evaluate(
            model, val_loader, metric, loss_fn, device
        )

        print(
            f"Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}  Val mAUROC: {mean_auc:.4f}"
        )

        is_best = mean_auc > best_auc
        if is_best:
            best_auc = mean_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_auc": best_auc,
            },
            is_best,
            output_dir,
        )

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} min. Best Val mAUROC: {best_auc:.4f}")

    # --------------------------------------------------------------------- #
    # Test evaluation with best model
    # --------------------------------------------------------------------- #
    best_state = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_state["model"])
    test_metric = MultilabelAUROC(num_labels=len(FINDINGS), average=None).to(device)
    _, test_per_class_auc, test_mean_auc = evaluate(
        model, test_loader, test_metric, loss_fn, device
    )
    print(f"Test mAUROC: {test_mean_auc:.4f}")

    # Save per-class results
    result_df = pd.DataFrame(
        {"Finding": FINDINGS, "AUROC": test_per_class_auc.round(4)}
    )
    result_df.to_csv(output_dir / "results.csv", index=False)
    print(f"Per-class AUROC saved to {output_dir/'results.csv'}")


if __name__ == "__main__":
    main()
