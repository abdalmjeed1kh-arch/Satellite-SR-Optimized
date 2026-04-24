"""
SatelliteDataset: PyTorch Dataset for loading satellite images.

Expects a directory layout where high-resolution (HR) images are stored
directly.  Low-resolution (LR) images are generated on-the-fly by
downscaling each HR image by *scale_factor*.

    data_root/
        ├── image_001.png
        ├── image_002.tif
        └── ...

Supported image extensions: .png, .jpg, .jpeg, .tif, .tiff.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _default_hr_transform(patch_size: int) -> Callable:
    """Return a transform that crops and converts an HR image to a tensor."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )


def _default_val_transform(patch_size: int) -> Callable:
    """Return a deterministic center-crop transform for validation."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ]
    )


class SatelliteDataset(Dataset):
    """Dataset that yields (lr_image, hr_image) pairs from a directory.

    Args:
        data_root: Path to the folder containing HR satellite images.
        scale_factor: Downscaling factor used to generate LR images.
        patch_size: Spatial size of the HR crop.  LR crops will have size
            ``patch_size // scale_factor``.
        split: ``"train"`` or ``"val"``; determines the augmentation applied.
        hr_transform: Optional custom transform applied to the HR numpy array
            before LR generation.  Receives a ``(H, W, C)`` uint8 ndarray and
            must return a ``torch.Tensor`` of shape ``(C, patch_size, patch_size)``.
        lr_transform: Optional custom transform applied to the LR numpy array.
            Receives a ``(H, W, C)`` uint8 ndarray and must return a
            ``torch.Tensor`` of shape ``(C, patch_size // scale_factor,
            patch_size // scale_factor)``.
    """

    def __init__(
        self,
        data_root: str | Path,
        scale_factor: int = 4,
        patch_size: int = 128,
        split: str = "train",
        hr_transform: Optional[Callable] = None,
        lr_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.split = split

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        self.image_paths = sorted(
            p
            for p in self.data_root.iterdir()
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.data_root}. "
                f"Supported extensions: {_SUPPORTED_EXTENSIONS}"
            )

        if split == "train":
            default_hr_tf = _default_hr_transform(patch_size)
        else:
            default_hr_tf = _default_val_transform(patch_size)

        self.hr_transform: Callable = hr_transform or default_hr_tf
        self.lr_transform: Callable = lr_transform or transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> np.ndarray:
        """Load an image as an RGB uint8 ndarray via OpenCV."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _downscale(self, hr_numpy: np.ndarray) -> np.ndarray:
        """Create a bicubic LR version of a uint8 HR ndarray."""
        h, w = hr_numpy.shape[:2]
        lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
        lr = cv2.resize(hr_numpy, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        return lr

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[index]
        hr_numpy = self._load_image(path)

        # Apply HR transform (crop / flip / to-tensor).
        hr_tensor: torch.Tensor = self.hr_transform(hr_numpy)

        # Convert back to numpy for downscaling (CHW float → HWC uint8).
        hr_numpy_cropped = (
            hr_tensor.permute(1, 2, 0).numpy() * 255.0
        ).astype(np.uint8)

        lr_numpy = self._downscale(hr_numpy_cropped)
        lr_tensor: torch.Tensor = self.lr_transform(lr_numpy)

        return lr_tensor, hr_tensor
