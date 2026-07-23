"""Etapas de pré-processamento das imagens."""

from __future__ import annotations

import numpy as np


def rgb_to_grayscale(images_rgb: np.ndarray) -> np.ndarray:
    """
    Converte imagens RGB para escala de cinza.
    Entrada esperada: (N, H, W, 3)
    Saída: (N, H, W)
    """
    if images_rgb.ndim != 4 or images_rgb.shape[-1] != 3:
        raise ValueError("images_rgb must have shape (N, H, W, 3).")

    images_float = images_rgb.astype(np.float32)
    gray = (
        0.299 * images_float[..., 0]
        + 0.587 * images_float[..., 1]
        + 0.114 * images_float[..., 2]
    )
    return gray.astype(np.float32)


def sobel_edges(gray_images: np.ndarray) -> np.ndarray:
    """
    Detecta bordas com Sobel em lote.
    Entrada: (N, H, W)
    Saída: (N, H, W)
    """
    if gray_images.ndim != 3:
        raise ValueError("gray_images must have shape (N, H, W).")

    padded = np.pad(gray_images, ((0, 0), (1, 1), (1, 1)), mode="edge")

    gx = (
        -1.0 * padded[:, :-2, :-2] + 1.0 * padded[:, :-2, 2:]
        -2.0 * padded[:, 1:-1, :-2] + 2.0 * padded[:, 1:-1, 2:]
        -1.0 * padded[:, 2:, :-2] + 1.0 * padded[:, 2:, 2:]
    )

    gy = (
        -1.0 * padded[:, :-2, :-2] - 2.0 * padded[:, :-2, 1:-1] - 1.0 * padded[:, :-2, 2:]
        + 1.0 * padded[:, 2:, :-2] + 2.0 * padded[:, 2:, 1:-1] + 1.0 * padded[:, 2:, 2:]
    )

    magnitude = np.sqrt(gx * gx + gy * gy)

    max_per_image = magnitude.reshape(magnitude.shape[0], -1).max(axis=1)
    max_per_image = np.where(max_per_image == 0, 1.0, max_per_image)
    magnitude = magnitude / max_per_image[:, None, None]

    return magnitude.astype(np.float32)
