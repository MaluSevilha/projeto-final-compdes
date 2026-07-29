#!/usr/bin/env python3
"""Profiling do pipeline Numba paralelo.

Atencao ao ler o resultado: cProfile nao enxerga dentro do codigo compilado,
so ve a chamada. Sem --warmup, o perfil mostra a compilacao JIT dominando o
tempo. Com --warmup, o perfil mostra o custo real de execucao, onde o
preprocessamento praticamente desaparece e sobra a carga do pickle.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from common.config import OUTPUTS_DIR
from experiments.numba_parallel.pipeline import process_pipeline
from experiments.numba_parallel.preprocessing import warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the parallel Numba CIFAR pipeline.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "profiling" / "numba_parallel_profile.prof",
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Compila antes de perfilar, para o perfil refletir a execucao e nao o compilador.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.warmup:
        used = warmup(args.threads)
        print(f"Warm-up concluido com {used} threads.")

    profiler = cProfile.Profile()
    profiler.enable()
    process_pipeline(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        threads=args.threads,
    )
    profiler.disable()

    profiler.dump_stats(str(args.output))
    print(f"Profile saved to: {args.output}")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
