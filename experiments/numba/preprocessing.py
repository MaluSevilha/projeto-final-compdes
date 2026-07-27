"""Numba implementation for grayscale + Sobel."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception as exc:  # pragma: no cover
    njit = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None


def _require_numba() -> None:
    if njit is None:
        raise RuntimeError("Numba is required for the numba version.") from _NUMBA_IMPORT_ERROR


if njit is not None:

    @njit(cache=True)
    def rgb_to_grayscale_single_numba(image_rgb):
        gray = np.empty((32, 32), dtype=np.float32)
        for i in range(32):
            for j in range(32):
                r = float(image_rgb[i, j, 0])
                g = float(image_rgb[i, j, 1])
                b = float(image_rgb[i, j, 2])
                gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
        return gray


    @njit(cache=True)
    def sobel_single_numba(gray_image):
        edges = np.empty((32, 32), dtype=np.float32)

        for i in range(32):
            for j in range(32):
                gx = 0.0
                gy = 0.0

                for di in (-1, 0, 1):
                    ii = i + di
                    if ii < 0:
                        ii = 0
                    elif ii > 31:
                        ii = 31

                    for dj in (-1, 0, 1):
                        jj = j + dj
                        if jj < 0:
                            jj = 0
                        elif jj > 31:
                            jj = 31

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

        max_val = 0.0
        for i in range(32):
            for j in range(32):
                if edges[i, j] > max_val:
                    max_val = edges[i, j]

        if max_val > 0.0:
            for i in range(32):
                for j in range(32):
                    edges[i, j] = edges[i, j] / max_val

        return edges


    @njit(cache=True)
    def process_images_numba(images_rgb):
        n = images_rgb.shape[0]
        grays = np.empty((n, 32, 32), dtype=np.float32)
        edges = np.empty((n, 32, 32), dtype=np.float32)

        for idx in range(n):
            gray = rgb_to_grayscale_single_numba(images_rgb[idx])
            edge = sobel_single_numba(gray)
            grays[idx] = gray
            edges[idx] = edge

        return grays, edges


def process_images(images_rgb: np.ndarray):
    _require_numba()

    if images_rgb.ndim != 4 or images_rgb.shape[-1] != 3:
        raise ValueError("images_rgb must have shape (N, 32, 32, 3).")

    return process_images_numba(images_rgb)
