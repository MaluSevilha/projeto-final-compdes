"""Pipeline paralelizado com multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from common.config import CIFAR10_DIR, CIFAR100_DIR
from common.dataset import load_dataset, reconstruct_images
from experiments.multiprocessing.preprocessing import process_images_parallel


@dataclass
class PipelineTimings:
    load_data_s: float = 0.0
    reconstruct_rgb_s: float = 0.0
    preprocess_s: float = 0.0
    total_s: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PipelineResult:
    images_rgb: np.ndarray
    images_gray: np.ndarray
    images_edges: np.ndarray
    labels: np.ndarray
    timings: PipelineTimings
    class_names: Optional[List[str]] = None


def now_s() -> float:
    return time.perf_counter()


def time_block(func, *args, **kwargs):
    start = now_s()
    result = func(*args, **kwargs)
    return result, now_s() - start


def get_data_dir(dataset: str) -> Path:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return CIFAR10_DIR
    if dataset == "cifar100":
        return CIFAR100_DIR
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def process_pipeline(
    dataset: str = "cifar10",
    split: str = "train",
    limit: int = 5000,
    data_dir: Optional[Path] = None,
    processes: Optional[int] = None,
) -> PipelineResult:
    total_start = now_s()
    timings = PipelineTimings()

    if data_dir is None:
        data_dir = get_data_dir(dataset)

    (images_flat, labels, class_names), timings.load_data_s = time_block(
        load_dataset, data_dir, dataset, split, limit
    )
    images_rgb, timings.reconstruct_rgb_s = time_block(reconstruct_images, images_flat)

    if processes is None:
        processes = max(1, mp.cpu_count() - 1)

    (images_gray, images_edges), timings.preprocess_s = time_block(
        process_images_parallel, images_rgb, processes
    )

    timings.total_s = now_s() - total_start
    return PipelineResult(images_rgb, images_gray, images_edges, labels, timings, class_names)
