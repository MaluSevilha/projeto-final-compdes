#!/usr/bin/env python3
"""
Generate outputs/benchmarks/scaling_sandbox.csv for the CIFAR-10 project.

Expected CSV schema:
n_images, baseline, vectorized, numba, multiprocessing, gpu

Dois modos de medicao do Numba, escolhidos por --numba-mode:

- hot (default): warm-up antes de medir, entao o tempo reflete a execucao do
  codigo ja compilado. E o regime principal do relatorio.
  Gera outputs/benchmarks/scaling_hot.csv.

- cold: cada ponto roda em subprocesso novo com NUMBA_CACHE_DIR unico, o que
  forca a recompilacao JIT em toda medicao. Serve para quantificar o custo fixo
  da compilacao. Gera outputs/benchmarks/scaling_numba_cold.csv.

Os volumes pedidos sao ajustados ao teto do split (o treino da CIFAR-10 tem
50.000 imagens), para nao registrar pontos que medem o mesmo volume.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics as stats
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd


def find_project_root() -> Path:
    """
    Finds the repository root by looking for the expected project folders.
    Works when running from the repo root, benchmark/, or a notebook.
    """
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]

    for p in candidates:
        if (p / "common").exists() and (p / "experiments").exists():
            return p

    raise RuntimeError(
        "Não consegui localizar a raiz do projeto. "
        "Abra o notebook dentro do repositório ou rode o script a partir dele."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------- utilities ----------

def load_callable(module_candidates: Sequence[str], attr_candidates: Sequence[str]) -> Callable[..., Any]:
    """Import the first callable found among the candidate modules/attributes."""
    errors: list[str] = []
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}: import failed ({exc})")
            continue

        for attr in attr_candidates:
            fn = getattr(module, attr, None)
            if callable(fn):
                return fn

        errors.append(f"{module_name}: none of {attr_candidates!r} found")

    joined = "\n - ".join(errors)
    raise ImportError(
        "Não consegui localizar a função de pipeline em nenhum módulo candidato.\n"
        f"Tentativas:\n - {joined}"
    )


def extract_total_seconds(result: Any) -> float:
    """Try several common result layouts used in the project."""
    timings = getattr(result, "timings", None)
    if timings is not None and hasattr(timings, "total_s"):
        return float(timings.total_s)

    for attr in ("total_s", "elapsed_s", "time_s", "duration_s"):
        if hasattr(result, attr):
            return float(getattr(result, attr))

    if isinstance(result, dict):
        for key in ("total_s", "elapsed_s", "time_s", "duration_s"):
            if key in result:
                return float(result[key])

    if isinstance(result, (tuple, list)) and result:
        for item in reversed(result):
            if isinstance(item, (int, float)):
                return float(item)

    raise TypeError(
        "Não consegui extrair o tempo total do resultado retornado pela pipeline."
    )


def call_pipeline(fn: Callable[..., Any], *, dataset: str, split: str, limit: int) -> Any:
    try:
        return fn(dataset=dataset, split=split, limit=limit)
    except TypeError:
        return fn(dataset, split, limit)


def run_version(
    fn: Callable[..., Any],
    *,
    dataset: str,
    split: str,
    limit: int,
    repeat: int,
    warmup: bool,
) -> float:
    """Return the mean total time in seconds for one version."""
    if warmup:
        try:
            call_pipeline(fn, dataset=dataset, split=split, limit=min(limit, 32))
        except Exception:
            pass

    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = call_pipeline(fn, dataset=dataset, split=split, limit=limit)
        end = time.perf_counter()

        try:
            total_s = extract_total_seconds(result)
        except Exception:
            total_s = end - start

        samples.append(float(total_s))

    return float(stats.mean(samples))


@dataclass(frozen=True)
class VersionSpec:
    name: str
    modules: tuple[str, ...]
    attrs: tuple[str, ...] = ("process_pipeline", "run_pipeline", "pipeline")


def build_versions() -> list[VersionSpec]:
    return [
        VersionSpec("baseline", ("experiments.baseline.pipeline",)),
        VersionSpec("vectorized", ("experiments.vectorized.pipeline", "experiments.numpy.pipeline")),
        VersionSpec("numba", ("experiments.numba.pipeline", "experiments.numba_impl.pipeline")),
        VersionSpec("multiprocessing", ("experiments.multiprocessing.pipeline", "experiments.parallel.pipeline")),
        VersionSpec("gpu", ("experiments.gpu.pipeline", "experiments.torch.pipeline")),
    ]


# ---------- Numba cold-run mode ----------

def run_numba_child(dataset: str, split: str, limit: int) -> int:
    """
    One cold Numba measurement in a fresh process.
    This includes JIT compilation because:
    - the process is new;
    - NUMBA_CACHE_DIR is unique for this run.
    """
    fn = load_callable(
        ("experiments.numba.pipeline", "experiments.numba_impl.pipeline"),
        ("process_pipeline", "run_pipeline", "pipeline"),
    )

    start = time.perf_counter()
    _ = call_pipeline(fn, dataset=dataset, split=split, limit=limit)
    end = time.perf_counter()

    payload = {"seconds": end - start}
    print(json.dumps(payload))
    return 0


def measure_numba_cold(
    *,
    dataset: str,
    split: str,
    limit: int,
) -> float:
    """
    Runs the Numba benchmark in a fresh subprocess with a unique cache dir,
    so the compilation cost is counted every time.
    """
    script_path = Path(__file__).resolve()

    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_cache_")

    cmd = [
        sys.executable,
        str(script_path),
        "--child-numba-run",
        "--dataset",
        dataset,
        "--split",
        split,
        "--limit",
        str(limit),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
            return float(payload["seconds"])
        except Exception:
            continue

    raise RuntimeError(
        "Não consegui ler o tempo do subprocesso do Numba.\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the scaling benchmark CSV.")
    parser.add_argument(
        "--numba-mode",
        choices=["hot", "cold"],
        default="hot",
        help="hot: mede o Numba já compilado (regime principal). "
             "cold: recompila em cada ponto, para medir o custo do JIT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path do CSV de saída. Default: scaling_hot.csv ou "
             "scaling_numba_cold.csv, conforme --numba-mode.",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="Nome do dataset usado pelo projeto.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split usado no benchmark.",
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[
            100, 500, 1000, 2000, 5000,
            10000, 15000, 20000, 30000, 40000, 50000,
        ],
        help="Lista de tamanhos de entrada a testar.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Número de repetições por tamanho.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Desativa warm-up antes da medição das versões não-Numba.",
    )
    parser.add_argument(
        "--child-numba-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # Child mode: only runs one cold Numba measurement and exits.
    if args.child_numba_run:
        if args.limit <= 0:
            raise ValueError("--limit precisa ser > 0 no modo child do Numba.")
        return run_numba_child(args.dataset, args.split, args.limit)

    # Load and validate the dataset once so we fail early if the project is not configured.
    from common.config import CIFAR10_DIR
    from common.dataset import load_dataset, reconstruct_images

    if not CIFAR10_DIR.exists():
        raise FileNotFoundError(f"Base CIFAR-10 não encontrada em: {CIFAR10_DIR}")

    if args.output is None:
        name = "scaling_hot.csv" if args.numba_mode == "hot" else "scaling_numba_cold.csv"
        args.output = PROJECT_ROOT / "outputs" / "benchmarks" / name

    # o split inteiro, para descobrir o teto real de imagens disponiveis
    images_flat, labels, class_names = load_dataset(
        CIFAR10_DIR,
        dataset=args.dataset,
        split=args.split,
        limit=None,
    )
    images_rgb = reconstruct_images(images_flat)
    ceiling = int(images_rgb.shape[0])

    counts = sorted({c for c in args.counts if 0 < c <= ceiling})
    if ceiling not in counts:
        counts.append(ceiling)
    if counts != sorted(set(args.counts)):
        print(f"Volumes ajustados ao teto do split '{args.split}' ({ceiling} imagens).")
    args.counts = counts

    print(f"Carregadas {ceiling} imagens para benchmark.")
    print(f"Modo do Numba: {args.numba_mode}")
    print(f"Classes: {class_names if class_names else 'n/d'}")

    # resolve as versoes e descarta as que nao executam nesta maquina
    # (o import pode funcionar e a execucao falhar, p.ex. torch sem CUDA)
    resolved: dict[str, Callable[..., Any]] = {}
    versions = []
    for spec in build_versions():
        try:
            fn = load_callable(spec.modules, spec.attrs)
            call_pipeline(fn, dataset=args.dataset, split=args.split, limit=2)
        except Exception as exc:  # noqa: BLE001
            print(f"AVISO: versao '{spec.name}' indisponivel, removendo do benchmark "
                  f"({type(exc).__name__}: {exc}).")
            continue
        resolved[spec.name] = fn
        versions.append(spec)

    if not versions:
        raise RuntimeError("Nenhuma versao disponivel para medir.")
    print(f"Versoes: {[v.name for v in versions]}")

    rows: list[dict[str, Any]] = []

    for n in args.counts:
        print(f"\nBenchmarking n_images = {n}")
        row: dict[str, Any] = {"n_images": n}

        for spec in versions:
            if spec.name == "numba" and args.numba_mode == "cold":
                # frio: processo novo + NUMBA_CACHE_DIR unico em cada medicao
                total_s = measure_numba_cold(
                    dataset=args.dataset,
                    split=args.split,
                    limit=n,
                )
            else:
                fn = resolved[spec.name]
                warmup = not args.no_warmup and spec.name in ("numba", "gpu")
                total_s = run_version(
                    fn,
                    dataset=args.dataset,
                    split=args.split,
                    limit=n,
                    repeat=args.repeat,
                    warmup=warmup,
                )

            row[spec.name] = total_s
            print(f"  {spec.name:<14} {total_s:.6f} s")

        rows.append(row)

    df = pd.DataFrame(rows, columns=["n_images", *[v.name for v in versions]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)

    print(f"\nCSV salvo em: {args.output}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())