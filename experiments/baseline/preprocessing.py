"""Baseline puro em Python: laços aninhados para grayscale e Sobel."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def rgb_to_grayscale_single(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.shape != (32, 32, 3):
        raise ValueError(f"Expected shape (32, 32, 3), got {image_rgb.shape}")

    gray = np.empty((32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            r = float(image_rgb[i, j, 0])
            g = float(image_rgb[i, j, 1])
            b = float(image_rgb[i, j, 2])
            gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def sobel_single(gray_image: np.ndarray) -> np.ndarray:
    if gray_image.shape != (32, 32):
        raise ValueError(f"Expected shape (32, 32), got {gray_image.shape}")

    edges = np.empty((32, 32), dtype=np.float32)

    for i in range(32):
        for j in range(32):
            gx = 0.0
            gy = 0.0

            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii = _clamp(i + di, 0, 31)
                    jj = _clamp(j + dj, 0, 31)
                    val = float(gray_image[ii, jj])

                    if di == -1 and dj == -1:
                        gx += -1.0 * val
                        gy += -1.0 * val
                    elif di == -1 and dj == 0:
                        gy += -2.0 * val
                    elif di == -1 and dj == 1:
                        gx += 1.0 * val
                        gy += -1.0 * val
                    elif di == 0 and dj == -1:
                        gx += -2.0 * val
                    elif di == 0 and dj == 1:
                        gx += 2.0 * val
                    elif di == 1 and dj == -1:
                        gx += -1.0 * val
                        gy += 1.0 * val
                    elif di == 1 and dj == 0:
                        gy += 2.0 * val
                    elif di == 1 and dj == 1:
                        gx += 1.0 * val
                        gy += 1.0 * val

            edges[i, j] = (gx * gx + gy * gy) ** 0.5

    max_val = float(edges.max())
    if max_val > 0:
        edges = edges / max_val
    return edges.astype(np.float32)


def process_images(images_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if images_rgb.ndim != 4 or images_rgb.shape[-1] != 3:
        raise ValueError("images_rgb must have shape (N, 32, 32, 3).")

    n = images_rgb.shape[0]
    grays = np.empty((n, 32, 32), dtype=np.float32)
    edges = np.empty((n, 32, 32), dtype=np.float32)

    for idx in range(n):
        gray = rgb_to_grayscale_single(images_rgb[idx])
        edge = sobel_single(gray)
        grays[idx] = gray
        edges[idx] = edge

    return grays, edges
