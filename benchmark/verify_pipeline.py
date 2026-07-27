#!/usr/bin/env python3
"""
Pipeline verifier for CIFAR experiments.

What it checks:
- loads a small sample from CIFAR;
- validates shapes and label mapping;
- runs the baseline and vectorized pipelines;
- compares outputs numerically;
- saves a visual preview of RGB / gray / edges for one sample.

Usage:
    python3 -m benchmark.verify_pipeline
    python3 -m benchmark.verify_pipeline --limit 20 --sample-index 0

Expected project structure:
- common/
- experiments/
- benchmark/
- data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import CIFAR10_DIR
from common.dataset import load_dataset, reconstruct_image
from experiments.baseline.pipeline import process_pipeline as run_baseline
from experiments.vectorized.pipeline import process_pipeline as run_vectorized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CIFAR pipeline correctness.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=20, help="Small subset for correctness checks.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/verification"))
    return parser.parse_args()


def summarize_basic(images_flat: np.ndarray, labels: np.ndarray, class_names) -> None:
    print("\n=== Basic dataset checks ===")
    print(f"Flat images shape: {images_flat.shape}")
    print(f"Labels shape:      {labels.shape}")
    print(f"Images dtype:      {images_flat.dtype}")
    print(f"Labels dtype:      {labels.dtype}")
    print(f"Min pixel:         {images_flat.min()}")
    print(f"Max pixel:         {images_flat.max()}")
    print(f"Mean pixel:        {images_flat.mean():.3f}")
    print(f"Std pixel:         {images_flat.std():.3f}")

    unique = np.unique(labels)
    print(f"Unique labels in subset: {unique.tolist()}")
    if class_names:
        print(f"Class count: {len(class_names)}")
        print("First 5 class names:", class_names[:5])


def compare_outputs(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray, atol: float = 1e-6) -> None:
    print(f"\n=== Comparing {name_a} vs {name_b} ===")
    print(f"{name_a} shape: {a.shape}")
    print(f"{name_b} shape: {b.shape}")

    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {name_a}={a.shape} vs {name_b}={b.shape}")

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"Max abs diff:  {diff.max():.8f}")
    print(f"Mean abs diff: {diff.mean():.8f}")

    if not np.allclose(a, b, atol=atol, rtol=0):
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        raise AssertionError(
            f"Outputs differ more than tolerance {atol}. "
            f"Worst index={idx}, {name_a}={a[idx]}, {name_b}={b[idx]}"
        )

    print("Result: OK (within tolerance)")


def validate_pipeline_outputs(result, expected_n: int) -> None:
    print("\n=== Pipeline output checks ===")
    assert result.images_rgb.shape[0] == expected_n, "Unexpected number of RGB images"
    assert result.images_gray.shape[0] == expected_n, "Unexpected number of gray images"
    assert result.images_edges.shape[0] == expected_n, "Unexpected number of edge images"

    assert result.images_rgb.shape[1:] == (32, 32, 3), "RGB shape should be (N, 32, 32, 3)"
    assert result.images_gray.shape[1:] == (32, 32), "Gray shape should be (N, 32, 32)"
    assert result.images_edges.shape[1:] == (32, 32), "Edges shape should be (N, 32, 32)"

    assert result.labels.shape[0] == expected_n, "Labels length mismatch"

    assert np.isfinite(result.images_gray).all(), "Gray image contains non-finite values"
    assert np.isfinite(result.images_edges).all(), "Edges image contains non-finite values"

    print("Shape and finiteness checks: OK")

    if result.images_gray.min() < -1e-6 or result.images_gray.max() > 255.0 + 1e-6:
        print("Warning: gray images outside [0, 255] range")
    if result.images_edges.min() < -1e-6:
        print("Warning: edge images contain negative values")

    print(f"Gray range:  [{result.images_gray.min():.4f}, {result.images_gray.max():.4f}]")
    print(f"Edges range: [{result.images_edges.min():.4f}, {result.images_edges.max():.4f}]")


def save_preview(result, labels, class_names, sample_index: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = result.images_rgb[sample_index]
    gray = result.images_gray[sample_index]
    edges = result.images_edges[sample_index]
    label = int(labels[sample_index])

    if class_names and 0 <= label < len(class_names):
        title = f"class {label}: {class_names[label]}"
    else:
        title = f"class {label}"

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(rgb.astype(np.uint8))
    axes[0].set_title(f"RGB\n{title}")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Gray")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Edges")
    axes[2].axis("off")

    fig.tight_layout()
    out_file = output_dir / f"sample_{sample_index}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview saved to: {out_file}")


def main() -> None:
    args = parse_args()

    # Load a small raw subset once for sanity checks.
    if args.dataset == "cifar10":
        data_dir = CIFAR10_DIR
    else:
        data_dir = Path("data") / "cifar-100-python"

    images_flat, labels, class_names = load_dataset(
        data_dir,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
    )
    summarize_basic(images_flat, labels, class_names)

    # Run both versions on the same subset.
    baseline = run_baseline(dataset=args.dataset, split=args.split, limit=args.limit)
    vectorized = run_vectorized(dataset=args.dataset, split=args.split, limit=args.limit)

    validate_pipeline_outputs(baseline, expected_n=args.limit)
    validate_pipeline_outputs(vectorized, expected_n=args.limit)

    # Compare results between versions.
    compare_outputs("RGB", baseline.images_rgb, "RGB", vectorized.images_rgb, atol=0)
    compare_outputs("Gray", baseline.images_gray, "Gray", vectorized.images_gray, atol=1e-4)
    compare_outputs("Edges", baseline.images_edges, "Edges", vectorized.images_edges, atol=1e-4)

    # Confirm label consistency.
    if not np.array_equal(baseline.labels, vectorized.labels):
        raise AssertionError("Labels differ between baseline and vectorized versions.")
    print("\nLabel consistency: OK")

    # Print timing summary.
    print("\n=== Timing summary ===")
    print(f"Baseline total:   {baseline.timings.total_s:.6f} s")
    print(f"Vectorized total: {vectorized.timings.total_s:.6f} s")

    save_preview(vectorized, labels, class_names, args.sample_index, args.output_dir)


if __name__ == "__main__":
    main()
