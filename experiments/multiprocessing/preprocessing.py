"""Processamento paralelo por imagem usando multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
from typing import List, Tuple

import numpy as np

from experiments.baseline.preprocessing import rgb_to_grayscale_single, sobel_single


def process_chunk(images_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray_list = []
    edge_list = []

    for image in images_chunk:
        gray = rgb_to_grayscale_single(image)
        edges = sobel_single(gray)
        gray_list.append(gray)
        edge_list.append(edges)

    return np.stack(gray_list, axis=0), np.stack(edge_list, axis=0)


def split_chunks(images_rgb: np.ndarray, n_chunks: int) -> List[np.ndarray]:
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive.")
    n = len(images_rgb)
    n_chunks = min(n_chunks, n)
    return [chunk for chunk in np.array_split(images_rgb, n_chunks) if len(chunk) > 0]


def process_images_parallel(images_rgb: np.ndarray, processes: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if images_rgb.ndim != 4 or images_rgb.shape[-1] != 3:
        raise ValueError("images_rgb must have shape (N, 32, 32, 3).")

    if processes is None:
        processes = max(1, mp.cpu_count() - 1)

    chunks = split_chunks(images_rgb, processes * 4)

    with mp.Pool(processes=processes) as pool:
        results = pool.map(process_chunk, chunks)

    grays = np.concatenate([r[0] for r in results], axis=0)
    edges = np.concatenate([r[1] for r in results], axis=0)
    return grays, edges
