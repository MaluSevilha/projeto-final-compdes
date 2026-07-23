#!/usr/bin/env python3
"""Roda o pipeline várias vezes e calcula médias/desvios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from pipeline import process_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the sequential CIFAR pipeline.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = []
    for i in range(args.runs):
        result = process_pipeline(dataset=args.dataset, split=args.split, limit=args.limit)
        rows.append({
            "run": i + 1,
            "load_data_s": result.timings.load_data_s,
            "reconstruct_rgb_s": result.timings.reconstruct_rgb_s,
            "grayscale_s": result.timings.grayscale_s,
            "edges_s": result.timings.edges_s,
            "total_s": result.timings.total_s,
            "images": int(result.images_rgb.shape[0]),
        })
        print(f"Run {i+1}/{args.runs}: total={result.timings.total_s:.6f}s")

    totals = [r["total_s"] for r in rows]
    print("\n=== Benchmark summary ===")
    print(f"Mean total time: {mean(totals):.6f}s")
    print(f"Std total time:  {stdev(totals) if len(totals) > 1 else 0.0:.6f}s")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
