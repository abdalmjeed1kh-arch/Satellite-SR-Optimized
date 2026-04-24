"""
evaluate.py — Evaluation script for SRResNet on satellite imagery.

Computes per-image and aggregate PSNR and SSIM scores over a test set.

Usage:
    python evaluate.py --data_root /path/to/images --checkpoint checkpoints/best_model.pth

Output:
    Prints per-image scores and a summary table, and optionally saves SR
    images to --output_dir.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from datasets import SatelliteDataset
from models import SRResNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SRResNet on satellite imagery")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing HR satellite images")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="HR patch size used during evaluation (default: 128)")
    parser.add_argument("--scale_factor", type=int, default=4,
                        help="Super-resolution scale factor (default: 4)")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a saved model checkpoint (.pth)")
    parser.add_argument("--num_features", type=int, default=64,
                        help="Feature channels matching the saved checkpoint (default: 64)")
    parser.add_argument("--num_residual_blocks", type=int, default=16,
                        help="Residual blocks matching the saved checkpoint (default: 16)")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size (default: 1)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes (default: 2)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="If provided, save SR images to this directory")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """Compute PSNR (dB) between two float images in [0, max_val]."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute mean SSIM between two float HWC images in [0, 1]."""
    score, _ = ssim_fn(
        img1,
        img2,
        data_range=1.0,
        channel_axis=2,
        full=True,
    )
    return float(score)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float tensor in [0, 1] to an HWC ndarray."""
    return t.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path | None,
) -> dict:
    model.eval()
    psnr_scores: list[float] = []
    ssim_scores: list[float] = []

    for idx, (lr_imgs, hr_imgs) in enumerate(loader):
        lr_imgs = lr_imgs.to(device)
        sr_imgs = model(lr_imgs)

        for i in range(sr_imgs.size(0)):
            sr_np = tensor_to_numpy(sr_imgs[i])
            hr_np = tensor_to_numpy(hr_imgs[i])

            img_psnr = psnr(sr_np, hr_np)
            img_ssim = ssim(sr_np, hr_np)
            psnr_scores.append(img_psnr)
            ssim_scores.append(img_ssim)

            global_idx = idx * loader.batch_size + i
            print(
                f"  Image {global_idx + 1:04d}  "
                f"PSNR: {img_psnr:6.2f} dB  "
                f"SSIM: {img_ssim:.4f}"
            )

            if output_dir is not None:
                sr_bgr = cv2.cvtColor(
                    (sr_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(
                    str(output_dir / f"sr_{global_idx + 1:04d}.png"), sr_bgr
                )

    return {
        "mean_psnr": float(np.mean(psnr_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
        "std_psnr": float(np.std(psnr_scores)),
        "std_ssim": float(np.std(ssim_scores)),
        "num_images": len(psnr_scores),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = SatelliteDataset(
        data_root=args.data_root,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        split="val",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SRResNet(
        scale_factor=args.scale_factor,
        num_features=args.num_features,
        num_residual_blocks=args.num_residual_blocks,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation ────────────────────────────────────────────────────────
    print(f"\nEvaluating on {len(dataset)} images ...\n")
    results = evaluate(model, loader, device, output_dir)

    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"  Images evaluated : {results['num_images']}")
    print(f"  Mean PSNR        : {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f} dB")
    print(f"  Mean SSIM        : {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f}")
    if output_dir:
        print(f"  SR images saved  : {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
