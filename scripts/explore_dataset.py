#!/usr/bin/env python3
"""Explora a CIFAR-10/CIFAR-100 e mostra um exemplo visual."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CIFAR10_DIR, CIFAR100_DIR
from dataset import batch_files, load_class_names, load_one_batch, reconstruct_image


def get_data_dir(dataset: str) -> Path:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return CIFAR10_DIR
    if dataset == "cifar100":
        return CIFAR100_DIR
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore CIFAR batches.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--batch", default=None, help="Batch filename to load.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--save-preview", type=Path, default=None)
    return parser.parse_args()


def summarize(images: np.ndarray, labels: np.ndarray, class_names):
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
            print(f"  {int(label):2d} -> {class_names[int(label)]:<12} : {int(count)}")
        else:
            print(f"  {int(label):2d} : {int(count)}")

    print("\nFirst 10 labels:")
    first_labels = labels[:10].tolist()
    if class_names:
        pretty = [class_names[int(x)] if 0 <= int(x) < len(class_names) else str(int(x)) for x in first_labels]
        print("  ", pretty)
    else:
        print("  ", first_labels)


def show_sample(images, labels, class_names, sample_index: int = 0, save_path: Path | None = None):
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
        print(f"Preview saved to: {save_path}")

    plt.show()


def main() -> None:
    args = parse_args()
    data_dir = get_data_dir(args.dataset)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Place the extracted CIFAR folder under the project's data/ directory."
        )

    class_names = load_class_names(data_dir, args.dataset)

    chosen_batch = args.batch
    if chosen_batch is None:
        chosen_batch = batch_files(args.dataset, args.split)[0]

    print(f"Dataset directory: {data_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Batch: {chosen_batch}")

    images, labels = load_one_batch(data_dir, args.dataset, chosen_batch)
    summarize(images, labels, class_names)
    show_sample(images, labels, class_names, args.sample_index, args.save_preview)


if __name__ == "__main__":
    main()
