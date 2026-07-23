#!/usr/bin/env python3
"""Executa o pipeline sequencial como baseline do projeto."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import process_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sequential CIFAR pipeline.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional override of the dataset directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional .npz output file.")
    return parser.parse_args()


def print_summary(result) -> None:
    n = int(result.images_rgb.shape[0])
    print("\n=== Resumo do pipeline sequencial ===")
    print(f"Imagens processadas: {n}")
    print(f"Shape RGB:   {result.images_rgb.shape}")
    print(f"Shape Gray:  {result.images_gray.shape}")
    print(f"Shape Edges: {result.images_edges.shape}")
    print(f"Total de classes (se disponível): {len(result.class_names) if result.class_names else 'N/D'}")

    print("\n--- Tempos por etapa ---")
    print(f"Load data:         {result.timings.load_data_s:.6f} s")
    print(f"Reconstruct RGB:   {result.timings.reconstruct_rgb_s:.6f} s")
    print(f"Grayscale:         {result.timings.grayscale_s:.6f} s")
    print(f"Edge detection:    {result.timings.edges_s:.6f} s")
    print(f"Total:             {result.timings.total_s:.6f} s")
    print(f"Throughput:        {n / result.timings.total_s:.2f} imagens/s")


def save_npz(output_path: Path, result) -> None:
    import numpy as np
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images_rgb=result.images_rgb,
        images_gray=result.images_gray,
        images_edges=result.images_edges,
        labels=result.labels,
        class_names=np.array(result.class_names if result.class_names is not None else [], dtype=object),
        timings=np.array([result.timings.to_dict()], dtype=object),
    )


def main() -> None:
    args = parse_args()
    result = process_pipeline(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        data_dir=args.data_dir,
    )
    print_summary(result)

    if args.output is not None:
        save_npz(args.output, result)
        print(f"\nSaída salva em: {args.output}")


if __name__ == "__main__":
    main()
