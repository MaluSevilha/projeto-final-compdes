#!/usr/bin/env python3
"""Compare baseline, Numba, NumPy, multiprocessing, and GPU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from experiments.baseline.pipeline import process_pipeline as run_baseline
from experiments.numba.pipeline import process_pipeline as run_numba
from experiments.vectorized.pipeline import process_pipeline as run_vectorized
from experiments.multiprocessing.pipeline import process_pipeline as run_mp
from experiments.gpu.pipeline import process_pipeline as run_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all CIFAR pipeline versions.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--csv", type=Path, default=Path("outputs/benchmarks/compare_versions.csv"))
    return parser.parse_args()


def benchmark_one(name: str, func, repeat: int = 3, **kwargs):
    times = []
    last = None
    for _ in range(repeat):
        last = func(**kwargs)
        times.append(last.timings.total_s)
    return {
        "version": name,
        "mean_total_s": mean(times),
        "std_total_s": stdev(times) if len(times) > 1 else 0.0,
        "load_data_s": last.timings.load_data_s,
        "reconstruct_rgb_s": getattr(last.timings, "reconstruct_rgb_s", 0.0),
        "preprocess_s": getattr(last.timings, "preprocess_s", 0.0),
        "grayscale_s": getattr(last.timings, "grayscale_s", 0.0),
        "edges_s": getattr(last.timings, "edges_s", 0.0),
        "n_images": int(last.images_rgb.shape[0]),
    }


def main() -> None:
    args = parse_args()
    rows = []

    rows.append(benchmark_one("baseline", run_baseline, repeat=args.repeat, dataset=args.dataset, split=args.split, limit=args.limit))
    rows.append(benchmark_one("numba", run_numba, repeat=args.repeat, dataset=args.dataset, split=args.split, limit=args.limit))
    rows.append(benchmark_one("vectorized", run_vectorized, repeat=args.repeat, dataset=args.dataset, split=args.split, limit=args.limit))
    rows.append(benchmark_one("multiprocessing", run_mp, repeat=args.repeat, dataset=args.dataset, split=args.split, limit=args.limit, processes=args.processes))

    try:
        rows.append(benchmark_one("gpu", run_gpu, repeat=args.repeat, dataset=args.dataset, split=args.split, limit=args.limit, device=args.device))
    except Exception as exc:
        print(f"GPU version skipped: {exc}")

    print("\n=== Comparison ===")
    for r in rows:
        print(f"{r['version']:<15} mean={r['mean_total_s']:.6f}s std={r['std_total_s']:.6f}s n={r['n_images']}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
