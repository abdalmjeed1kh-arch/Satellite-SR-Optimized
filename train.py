"""
train.py — Training script for SRResNet on satellite imagery.

Usage:
    python train.py --data_root /path/to/images --epochs 100

Key design choices:
  - Loss:      Pixel-wise MSE between the SR output and the HR ground truth.
  - Optimizer: Adam with a configurable learning rate.
  - Scheduler: ReduceLROnPlateau halves the LR when validation loss stalls.
  - Checkpointing: Best model weights are saved to --checkpoint_dir.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets import SatelliteDataset
from models import SRResNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRResNet for satellite SR")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing HR satellite images")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="HR patch size used during training (default: 128)")
    parser.add_argument("--scale_factor", type=int, default=4,
                        help="Super-resolution scale factor (default: 4)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data used for validation (default: 0.1)")

    # Model
    parser.add_argument("--num_features", type=int, default=64,
                        help="Feature channels in residual blocks (default: 64)")
    parser.add_argument("--num_residual_blocks", type=int, default=16,
                        help="Number of residual blocks (default: 16)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial Adam learning rate (default: 1e-4)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")

    # I/O
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for saving model checkpoints (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume training from")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    start = time.time()

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(loader):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            print(
                f"[Epoch {epoch}/{total_epochs}] "
                f"Step {batch_idx + 1}/{len(loader)}  "
                f"Loss: {loss.item():.6f}  "
                f"Elapsed: {elapsed:.1f}s"
            )

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for lr_imgs, hr_imgs in loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        sr_imgs = model(lr_imgs)
        total_loss += criterion(sr_imgs, hr_imgs).item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_dataset = SatelliteDataset(
        data_root=args.data_root,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        split="train",
    )
    val_dataset = SatelliteDataset(
        data_root=args.data_root,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        split="val",
    )

    # Use a fixed portion of the dataset for validation.
    val_size = max(1, int(len(val_dataset) * args.val_split))
    val_dataset, _ = random_split(val_dataset, [val_size, len(val_dataset) - val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = SRResNet(
        scale_factor=args.scale_factor,
        num_features=args.num_features,
        num_residual_blocks=args.num_residual_blocks,
    ).to(device)

    # ── Loss and optimiser ───────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1})")

    # ── Checkpoint directory ─────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

        # Save periodic checkpoint every 10 epochs.
        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved periodic checkpoint → {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
