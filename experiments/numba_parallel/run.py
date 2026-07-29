#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.numba_parallel.pipeline import process_pipeline
from experiments.numba_parallel.preprocessing import max_threads, warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel Numba CIFAR pipeline.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Numero de threads do Numba (default: todas as disponiveis).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Compila o JIT antes de medir, para o tempo refletir so a execucao.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Teto de threads do Numba neste processo: {max_threads()}")
    if args.warmup:
        used = warmup(args.threads)
        print(f"Warm-up concluido com {used} threads (JIT ja compilado).")

    result = process_pipeline(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        data_dir=args.data_dir,
        threads=args.threads,
    )

    n = int(result.images_rgb.shape[0])
    print("\n=== Resumo do pipeline Numba paralelo ===")
    print(f"Imagens processadas: {n}")
    print(f"Threads usadas:      {result.threads}")
    print(f"Shape RGB:   {result.images_rgb.shape}")
    print(f"Shape Gray:  {result.images_gray.shape}")
    print(f"Shape Edges: {result.images_edges.shape}")
    print(f"Total de classes (se disponivel): {len(result.class_names) if result.class_names else 'N/D'}")

    print("\n--- Tempos por etapa ---")
    print(f"Load data:         {result.timings.load_data_s:.6f} s")
    print(f"Reconstruct RGB:   {result.timings.reconstruct_rgb_s:.6f} s")
    print(f"Preprocess:        {result.timings.preprocess_s:.6f} s")
    print(f"Total:             {result.timings.total_s:.6f} s")
    print(f"Throughput:        {n / result.timings.total_s:.2f} imagens/s")

    if not args.warmup:
        print(
            "\nNota: sem --warmup, o tempo de preprocess inclui a compilacao JIT "
            "da primeira chamada."
        )


if __name__ == "__main__":
    main()
