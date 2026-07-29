#!/usr/bin/env python3
"""Escalonamento do Numba paralelo contra NumPy, Numba serial e GPU.

Mede tudo em **regime quente** (warm-up antes de medir), do mesmo jeito que
scaling_hot.csv, para que as curvas sejam comparaveis e possam ser unidas na
analise do notebook.

Pode ser executado de qualquer lugar:

    python3 -m benchmark.scaling_parallel        # da raiz do projeto
    python3 scaling_parallel.py                  # de dentro de benchmark/
    !python scaling_parallel.py --repeat 5        # de um notebook em benchmark/

Saidas (em outputs/benchmarks/):
- scaling_parallel.csv         formato largo, tempo medio por versao (mesmo
                               schema de scaling_hot.csv, facil de concatenar)
- scaling_parallel_detail.csv  formato longo, com desvio padrao, minimo e threads
- threads_parallel.csv         varredura de numero de threads no maior volume

Exemplos:
    python3 scaling_parallel.py
    python3 scaling_parallel.py --repeat 5 --include-slow
    python3 scaling_parallel.py --counts 1000 10000 50000 --threads 4
    python3 scaling_parallel.py --no-thread-sweep
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd


# --------------------------------------------------------------------------
# raiz do projeto: precisa entrar no sys.path ANTES de importar experiments.*
# --------------------------------------------------------------------------

def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "common").exists() and (p / "experiments").exists():
            return p

    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "common").exists() and (p / "experiments").exists():
            return p

    raise RuntimeError(
        "Nao consegui localizar a raiz do projeto (esperava encontrar common/ e experiments/)."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCH_DIR = PROJECT_ROOT / "outputs" / "benchmarks"

DEFAULT_COUNTS = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000]

# ordem de exibicao e de colunas no CSV largo
FAST_VERSIONS = ["vectorized", "numba", "numba_parallel", "gpu"]
SLOW_VERSIONS = ["baseline", "multiprocessing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Escalonamento: numba_parallel vs vectorized vs numba vs gpu."
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=DEFAULT_COUNTS,
        help="Volumes a medir. Valores acima do teto do split sao ajustados.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads do Numba paralelo na curva principal (default: todas).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device do PyTorch.")
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Inclui baseline e multiprocessing (fica muito mais lento).",
    )
    parser.add_argument(
        "--no-thread-sweep",
        action="store_true",
        help="Nao roda a varredura de numero de threads.",
    )
    parser.add_argument(
        "--sweep-threads",
        nargs="+",
        type=int,
        default=None,
        help="Lista de threads da varredura (default: 1, 2, 4, ... ate o teto).",
    )
    parser.add_argument("--outdir", type=Path, default=BENCH_DIR)
    return parser.parse_args()


# --------------------------------------------------------------------------
# medicao
# --------------------------------------------------------------------------

def measure(
    fn: Callable[..., Any],
    repeat: int,
    warmup_limit: int = 32,
    **kwargs: Any,
) -> dict[str, float]:
    """Roda a pipeline `repeat` vezes em regime quente e agrega os tempos."""
    warm_kwargs = dict(kwargs)
    warm_kwargs["limit"] = min(int(kwargs.get("limit", warmup_limit)), warmup_limit)
    fn(**warm_kwargs)  # warm-up: paga JIT, alocacao de contexto CUDA, pool de threads

    samples: list[float] = []
    for _ in range(repeat):
        result = fn(**kwargs)
        samples.append(float(result.timings.total_s))

    return {
        "mean_s": stats.mean(samples),
        "std_s": stats.stdev(samples) if len(samples) > 1 else 0.0,
        "min_s": min(samples),
        "n_images": int(result.images_rgb.shape[0]),
    }


def build_runners(args: argparse.Namespace) -> dict[str, Callable[..., Any]]:
    """Resolve as pipelines disponiveis, pulando as que faltam dependencia."""
    from experiments.vectorized.pipeline import process_pipeline as run_vectorized

    runners: dict[str, Callable[..., Any]] = {"vectorized": run_vectorized}

    try:
        from experiments.numba.pipeline import process_pipeline as run_numba

        runners["numba"] = run_numba
    except Exception as exc:  # noqa: BLE001
        print(f"AVISO: versao numba indisponivel ({exc}).")

    try:
        from experiments.numba_parallel.pipeline import process_pipeline as run_numba_par
        from experiments.numba_parallel.preprocessing import max_threads

        def run_numba_parallel(**kwargs: Any):
            return run_numba_par(threads=args.threads, **kwargs)

        runners["numba_parallel"] = run_numba_parallel
        print(f"Teto de threads do Numba: {max_threads()} (usando: {args.threads or 'todas'})")
    except Exception as exc:  # noqa: BLE001
        print(f"AVISO: versao numba_parallel indisponivel ({exc}).")

    try:
        from experiments.gpu.pipeline import process_pipeline as run_gpu

        def run_gpu_wrapped(**kwargs: Any):
            return run_gpu(device=args.device, **kwargs)

        runners["gpu"] = run_gpu_wrapped
    except Exception as exc:  # noqa: BLE001
        print(f"AVISO: versao gpu indisponivel ({exc}).")

    if args.include_slow:
        from experiments.baseline.pipeline import process_pipeline as run_baseline
        from experiments.multiprocessing.pipeline import process_pipeline as run_mp

        runners["baseline"] = run_baseline
        runners["multiprocessing"] = run_mp

    return runners


def probe_runners(
    runners: dict[str, Callable[..., Any]],
    dataset: str,
    split: str,
) -> dict[str, Callable[..., Any]]:
    """Roda cada versao uma vez com 2 imagens e descarta as que nao executam.

    O import pode funcionar e a execucao falhar por falta de runtime (torch
    instalado sem CUDA, por exemplo). Melhor descobrir isso agora que no meio
    de um benchmark de meia hora.
    """
    alive: dict[str, Callable[..., Any]] = {}
    for name, fn in runners.items():
        try:
            fn(dataset=dataset, split=split, limit=2)
        except Exception as exc:  # noqa: BLE001
            print(f"AVISO: '{name}' nao executou, removendo do benchmark "
                  f"({type(exc).__name__}: {exc}).")
            continue
        alive[name] = fn
    return alive


def dataset_ceiling(dataset: str, split: str) -> int:
    """Quantas imagens o split realmente tem."""
    from common.config import CIFAR10_DIR, CIFAR100_DIR
    from common.dataset import load_dataset

    data_dir = CIFAR10_DIR if dataset == "cifar10" else CIFAR100_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Base nao encontrada em: {data_dir}")

    images_flat, _, _ = load_dataset(data_dir, dataset=dataset, split=split, limit=None)
    return int(images_flat.shape[0])


def clip_counts(counts: list[int], ceiling: int) -> list[int]:
    """Remove duplicatas e valores acima do teto, garantindo o teto na lista."""
    kept = sorted({c for c in counts if 0 < c <= ceiling})
    if ceiling not in kept:
        kept.append(ceiling)
    return kept


# --------------------------------------------------------------------------
# varredura de threads
# --------------------------------------------------------------------------

def default_sweep(hard_max: int) -> list[int]:
    values = []
    t = 1
    while t <= hard_max:
        values.append(t)
        t *= 2
    if hard_max not in values:
        values.append(hard_max)
    return values


def thread_sweep(args: argparse.Namespace, n_images: int) -> Optional[pd.DataFrame]:
    try:
        from experiments.numba_parallel.pipeline import process_pipeline as run_numba_par
        from experiments.numba_parallel.preprocessing import max_threads
    except Exception as exc:  # noqa: BLE001
        print(f"\nVarredura de threads pulada: numba_parallel indisponivel ({exc}).")
        return None

    hard_max = max_threads()
    sweep = args.sweep_threads or default_sweep(hard_max)
    sweep = sorted({min(max(1, t), hard_max) for t in sweep})

    print(f"\n=== Varredura de threads com {n_images} imagens ===")
    rows = []
    for t in sweep:
        res = measure(
            lambda **kw: run_numba_par(threads=t, **kw),
            repeat=args.repeat,
            dataset=args.dataset,
            split=args.split,
            limit=n_images,
        )
        rows.append({"threads": t, "n_images": res["n_images"], **{
            k: v for k, v in res.items() if k != "n_images"
        }})
        print(f"  {t:>3} thread(s)  {res['mean_s']:.4f} s")

    df = pd.DataFrame(rows)
    base = float(df.loc[df["threads"] == df["threads"].min(), "mean_s"].iloc[0])
    df["speedup_vs_min_threads"] = base / df["mean_s"]
    df["efficiency"] = df["speedup_vs_min_threads"] / (df["threads"] / df["threads"].min())
    return df


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Raiz do projeto : {PROJECT_ROOT}")
    print(f"CPUs visiveis   : {os.cpu_count()}")

    ceiling = dataset_ceiling(args.dataset, args.split)
    counts = clip_counts(args.counts, ceiling)
    print(f"Teto do split '{args.split}': {ceiling} imagens")
    print(f"Volumes a medir : {counts}")

    runners = probe_runners(build_runners(args), args.dataset, args.split)
    order = [v for v in (SLOW_VERSIONS + FAST_VERSIONS) if v in runners]
    if not order:
        raise RuntimeError("Nenhuma versao disponivel para medir.")
    print(f"Versoes         : {order}")
    print(f"Repeticoes      : {args.repeat} (todas em regime quente, com warm-up)\n")

    detail_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for n in counts:
        print(f"n_images = {n}")
        wide: dict[str, Any] = {"n_images": n}

        for name in order:
            res = measure(
                runners[name],
                repeat=args.repeat,
                dataset=args.dataset,
                split=args.split,
                limit=n,
            )
            wide[name] = round(res["mean_s"], 6)
            detail_rows.append({
                "n_images": res["n_images"],
                "version": name,
                "mean_s": res["mean_s"],
                "std_s": res["std_s"],
                "min_s": res["min_s"],
                "repeat": args.repeat,
                "threads": args.threads if name == "numba_parallel" else None,
            })
            print(f"  {name:<16} {res['mean_s']:>10.6f} s  (std {res['std_s']:.6f})")

        wide_rows.append(wide)
        print()

    df_wide = pd.DataFrame(wide_rows, columns=["n_images", *order])
    df_detail = pd.DataFrame(detail_rows)

    wide_path = args.outdir / "scaling_parallel.csv"
    detail_path = args.outdir / "scaling_parallel_detail.csv"
    df_wide.to_csv(wide_path, index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
    df_detail.to_csv(detail_path, index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)

    print("=== Tempo total (s) por versao x volume ===")
    print(df_wide.to_string(index=False))

    # leitura rapida no maior volume
    last = df_wide.iloc[-1]
    n_max = int(last["n_images"])
    print(f"\n=== Leitura com {n_max} imagens ===")
    for name in sorted(order, key=lambda v: last[v]):
        print(f"  {name:<16} {last[name]:>10.6f} s   {1e6 * last[name] / n_max:>7.2f} us/img")

    if "numba" in order and "numba_parallel" in order:
        print(f"\nnumba_parallel vs numba serial : {last['numba'] / last['numba_parallel']:.2f}x")
    if "vectorized" in order and "numba_parallel" in order:
        print(f"numba_parallel vs vectorized   : {last['vectorized'] / last['numba_parallel']:.2f}x")
    if "gpu" in order and "numba_parallel" in order:
        ratio = last["numba_parallel"] / last["gpu"]
        if ratio > 1:
            print(f"numba_parallel vs gpu          : {ratio:.2f}x do tempo da GPU (GPU ainda na frente)")
        else:
            print(f"numba_parallel vs gpu          : {1 / ratio:.2f}x mais rapido que a GPU")

    print(f"\nCSV salvo em: {wide_path}")
    print(f"CSV salvo em: {detail_path}")

    if not args.no_thread_sweep:
        df_threads = thread_sweep(args, n_max)
        if df_threads is not None:
            threads_path = args.outdir / "threads_parallel.csv"
            df_threads.to_csv(threads_path, index=False, float_format="%.6f")
            print("\n=== Escalabilidade por numero de threads ===")
            print(df_threads.to_string(index=False))
            print(f"\nCSV salvo em: {threads_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
