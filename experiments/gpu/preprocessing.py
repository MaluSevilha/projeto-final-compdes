"""Versão GPU com PyTorch/CUDA para grayscale e Sobel."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for the GPU version but is not available.") from _TORCH_IMPORT_ERROR


def _get_device(device: str | None = None):
    _require_torch()
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def rgb_to_grayscale_torch(images_rgb: np.ndarray, device: str | None = "auto"):
    _require_torch()
    dev = _get_device(device)
    x = torch.from_numpy(images_rgb).to(dev).permute(0, 3, 1, 2).float()
    gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
    return gray


def sobel_edges_torch(gray_images):
    _require_torch()
    if gray_images.ndim != 4 or gray_images.shape[1] != 1:
        raise ValueError("gray_images must have shape (N, 1, H, W).")

    device = gray_images.device

    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0],
          [-2.0, 0.0, 2.0],
          [-1.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0],
          [ 0.0,  0.0,  0.0],
          [ 1.0,  2.0,  1.0]]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    padded = F.pad(gray_images, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)
    gy = F.conv2d(padded, kernel_y)
    magnitude = torch.sqrt(gx * gx + gy * gy)

    flat = magnitude.flatten(1)
    max_per_image = flat.max(dim=1).values.clamp_min(1e-12)
    magnitude = magnitude / max_per_image.view(-1, 1, 1, 1)
    return magnitude


def process_images_gpu(images_rgb: np.ndarray, device: str | None = "auto") -> Tuple[np.ndarray, np.ndarray]:
    _require_torch()
    gray = rgb_to_grayscale_torch(images_rgb, device=device)
    edges = sobel_edges_torch(gray)
    gray_np = gray.squeeze(1).detach().cpu().numpy().astype(np.float32)
    edges_np = edges.squeeze(1).detach().cpu().numpy().astype(np.float32)
    return gray_np, edges_np
