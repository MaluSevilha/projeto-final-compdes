#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from experiments.numba.pipeline import process_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the Numba CIFAR pipeline.")
    parser.add_argument("--output", type=Path, default=Path("outputs/profiling/numba_profile.prof"))
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    process_pipeline(dataset=args.dataset, split=args.split, limit=args.limit)
    profiler.disable()

    profiler.dump_stats(str(args.output))
    print(f"Profile saved to: {args.output}")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
