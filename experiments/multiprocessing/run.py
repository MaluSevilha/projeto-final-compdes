#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.multiprocessing.pipeline import process_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiprocessing CIFAR pipeline.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = process_pipeline(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        data_dir=args.data_dir,
        processes=args.processes,
    )

    n = int(result.images_rgb.shape[0])
    print("\n=== Resumo do pipeline multiprocessing ===")
    print(f"Imagens processadas: {n}")
    print(f"Shape RGB:   {result.images_rgb.shape}")
    print(f"Shape Gray:  {result.images_gray.shape}")
    print(f"Shape Edges: {result.images_edges.shape}")
    print(f"Total de classes (se disponível): {len(result.class_names) if result.class_names else 'N/D'}")

    print("\n--- Tempos por etapa ---")
    print(f"Load data:         {result.timings.load_data_s:.6f} s")
    print(f"Reconstruct RGB:   {result.timings.reconstruct_rgb_s:.6f} s")
    print(f"Preprocess:        {result.timings.preprocess_s:.6f} s")
    print(f"Total:             {result.timings.total_s:.6f} s")
    print(f"Throughput:        {n / result.timings.total_s:.2f} imagens/s")


if __name__ == "__main__":
    main()
