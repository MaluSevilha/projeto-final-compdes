#!/usr/bin/env python3
"""
Explore CIFAR-10/CIFAR-100 Python batches.

What this script does:
- locates the dataset inside the project data directory;
- loads one batch from the Python version of CIFAR;
- prints shapes, keys, and label information;
- reconstructs and displays one sample image;
- optionally saves the preview image to disk.

Expected project structure:

project-root/
├── data/
│   └── cifar-10-batches-py/
│       ├── batches.meta
│       ├── ...
│       └── test_batch
└── scripts/
    └── explore_dataset.py

Usage:
    python scripts/explore_dataset.py
    python scripts/explore_dataset.py --dataset cifar100
    python scripts/explore_dataset.py --batch data_batch_1 --save-preview outputs/sample.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def project_root() -> Path:
    """Return the project root assuming this file lives in scripts/."""
    return Path(__file__).resolve().parent.parent


def default_dataset_dir(dataset: str) -> Path:
    """Return the default dataset folder inside ./data."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        return project_root() / "data" / "cifar-10-batches-py"
    if dataset == "cifar100":
        return project_root() / "data" / "cifar-100-python"
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def unpickle(path: Path) -> Dict[str, Any]:
    """Load a CIFAR pickle file with latin1 encoding."""
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def labels_key(dataset: str) -> str:
    """Return the label key used by the chosen dataset."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        return "labels"
    if dataset == "cifar100":
        return "fine_labels"
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def meta_file(dataset: str) -> str:
    """Return the metadata filename for the chosen dataset."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        return "batches.meta"
    if dataset == "cifar100":
        return "meta"
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def batch_files(dataset: str, split: str) -> List[str]:
    """Return the standard batch filenames for the chosen dataset and split."""
    dataset = dataset.lower()
    split = split.lower()

    if dataset == "cifar10":
        if split == "train":
            return [f"data_batch_{i}" for i in range(1, 6)]
        if split == "test":
            return ["test_batch"]
        raise ValueError("split must be 'train' or 'test'")

    if dataset == "cifar100":
        if split == "train":
            return ["train"]
        if split == "test":
            return ["test"]
        raise ValueError("split must be 'train' or 'test'")

    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def load_class_names(data_dir: Path, dataset: str) -> Optional[List[str]]:
    """Load class names from the metadata file if available."""
    path = data_dir / meta_file(dataset)
    if not path.exists():
        return None

    meta = unpickle(path)

    if dataset.lower() == "cifar10":
        names = meta.get("label_names")
    else:
        names = meta.get("fine_label_names")

    if not names:
        return None

    return [
        n.decode("utf-8") if isinstance(n, bytes) else str(n)
        for n in names
    ]


def load_one_batch(data_dir: Path, dataset: str, batch_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one CIFAR batch and return (flat_images, labels)."""
    path = data_dir / batch_name
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {path}")

    batch = unpickle(path)
    key = labels_key(dataset)

    images = np.asarray(batch["data"], dtype=np.uint8)
    labels = np.asarray(batch[key], dtype=np.int64)
    return images, labels


def reconstruct_image(flat_image: np.ndarray) -> np.ndarray:
    """
    Reconstruct one CIFAR image from (3072,) into (32, 32, 3).
    CIFAR stores channels in RGB blocks.
    """
    if flat_image.shape != (3072,):
        raise ValueError(f"Expected shape (3072,), got {flat_image.shape}")

    return flat_image.reshape(3, 32, 32).transpose(1, 2, 0)


def summarize_batch(images: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]]) -> None:
    """Print useful information about the loaded batch."""
    print("\n=== CIFAR batch summary ===")
    print(f"Images array shape: {images.shape}")
    print(f"Labels array shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Number of images: {len(images)}")

    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution in this batch:")
    for label, count in zip(unique, counts):
        if class_names and 0 <= int(label) < len(class_names):
            name = class_names[int(label)]
            print(f"  {int(label):2d} -> {name:<12} : {int(count)}")
        else:
            print(f"  {int(label):2d} : {int(count)}")

    print("\nFirst 10 labels:")
    first_labels = labels[:10].tolist()
    if class_names:
        pretty = [class_names[int(x)] if 0 <= int(x) < len(class_names) else str(int(x)) for x in first_labels]
        print("  ", pretty)
    else:
        print("  ", first_labels)


def show_sample_image(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]],
    sample_index: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """Display and optionally save one reconstructed sample image."""
    if not (0 <= sample_index < len(images)):
        raise IndexError(f"sample_index must be between 0 and {len(images) - 1}")

    image = reconstruct_image(images[sample_index])
    label = int(labels[sample_index])

    if class_names and 0 <= label < len(class_names):
        title = f"Sample {sample_index} - class {label}: {class_names[label]}"
    else:
        title = f"Sample {sample_index} - class {label}"

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nPreview saved to: {save_path}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore CIFAR Python batches.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--batch", default=None, help="Batch filename to load. If omitted, the first standard batch is used.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to the dataset folder. Defaults to ./data/<dataset-folder>.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the image to preview.")
    parser.add_argument("--save-preview", type=Path, default=None, help="Optional path to save the preview image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir if args.data_dir is not None else default_dataset_dir(args.dataset)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Place the extracted CIFAR folder under the project's data/ directory."
        )

    classes = load_class_names(data_dir, args.dataset)

    chosen_batch = args.batch
    if chosen_batch is None:
        chosen_batch = batch_files(args.dataset, args.split)[0]

    print(f"Dataset directory: {data_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Batch: {chosen_batch}")

    images, labels = load_one_batch(data_dir, args.dataset, chosen_batch)
    summarize_batch(images, labels, classes)
    show_sample_image(
        images=images,
        labels=labels,
        class_names=classes,
        sample_index=args.sample_index,
        save_path=args.save_preview,
    )


if __name__ == "__main__":
    main()
