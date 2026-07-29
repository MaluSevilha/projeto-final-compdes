"""Compara baseline, NumPy (vetorizada), Numba, multiprocessing e GPU.

Pode ser executado de qualquer lugar:

    python3 -m benchmark.compare_versions          # da raiz do projeto
    python3 compare_versions.py                    # de dentro de benchmark/
    !python compare_versions.py --limit 20000      # de um notebook em benchmark/

Medicao:
- por padrao roda um warm-up antes de medir Numba e GPU (regime "quente"),
  que e o cenario de um servico que processa varios lotes no mesmo processo;
- com --cold-numba, o Numba e medido em subprocesso novo e cache limpo,
  incluindo o custo de compilacao JIT em cada medicao.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable


# --------------------------------------------------------------------------
# raiz do projeto: precisa entrar no sys.path ANTES de importar experiments.*
# --------------------------------------------------------------------------

def find_project_root() -> Path:
    """Acha a raiz do repo procurando as pastas common/ e experiments/."""
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

DEFAULT_CSV = PROJECT_ROOT / "outputs" / "benchmarks" / "compare_versions.csv"

STAGE_FIELDS = ("load_data_s", "reconstruct_rgb_s", "preprocess_s", "grayscale_s", "edges_s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all CIFAR pipeline versions.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Nao roda warm-up antes de medir Numba/GPU.",
    )
    parser.add_argument(
        "--cold-numba",
        action="store_true",
        help="Mede o Numba em subprocesso novo com cache limpo (inclui a compilacao JIT).",
    )
    parser.add_argument("--child-numba-run", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_csv_path(raw: Path) -> Path:
    """Caminho relativo passado na CLI e ancorado na raiz do projeto, nao no cwd."""
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw)


# --------------------------------------------------------------------------
# medicao
# --------------------------------------------------------------------------

def benchmark_one(
    name: str,
    func: Callable[..., Any],
    repeat: int = 3,
    warmup: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Roda a pipeline `repeat` vezes e agrega tempos totais e por etapa."""
    if warmup:
        warm_kwargs = dict(kwargs)
        warm_kwargs["limit"] = min(int(kwargs.get("limit", 32)), 32)
        try:
            func(**warm_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] warm-up falhou ({exc}); seguindo sem warm-up.")

    times: list[float] = []
    stages: dict[str, list[float]] = {k: [] for k in STAGE_FIELDS}
    last = None

    for _ in range(repeat):
        last = func(**kwargs)
        times.append(last.timings.total_s)
        for field in STAGE_FIELDS:
            stages[field].append(float(getattr(last.timings, field, 0.0) or 0.0))

    row: dict[str, Any] = {
        "version": name,
        "mean_total_s": mean(times),
        "std_total_s": stdev(times) if len(times) > 1 else 0.0,
        "min_total_s": min(times),
        "max_total_s": max(times),
    }
    # etapas na mesma base do mean_total_s (media das repeticoes, nao a ultima)
    row.update({field: mean(vals) for field, vals in stages.items()})
    row["n_images"] = int(last.images_rgb.shape[0])
    row["repeat"] = repeat
    return row


# --------------------------------------------------------------------------
# modo Numba frio (subprocesso + cache novo)
# --------------------------------------------------------------------------

def run_numba_child(dataset: str, split: str, limit: int) -> int:
    from experiments.numba.pipeline import process_pipeline as run_numba

    start = time.perf_counter()
    result = run_numba(dataset=dataset, split=split, limit=limit)
    wall = time.perf_counter() - start

    payload = {
        "total_s": float(result.timings.total_s),
        "wall_s": wall,
        "load_data_s": float(result.timings.load_data_s),
        "reconstruct_rgb_s": float(getattr(result.timings, "reconstruct_rgb_s", 0.0)),
        "preprocess_s": float(getattr(result.timings, "preprocess_s", 0.0)),
        "n_images": int(result.images_rgb.shape[0]),
    }
    print(json.dumps(payload))
    return 0


def benchmark_numba_cold(dataset: str, split: str, limit: int, repeat: int) -> dict[str, Any]:
    times: list[float] = []
    payload: dict[str, Any] = {}

    for _ in range(repeat):
        env = os.environ.copy()
        env["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_cache_")
        proc = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()), "--child-numba-run",
                "--dataset", dataset, "--split", split, "--limit", str(limit),
            ],
            capture_output=True, text=True, env=env, check=True,
        )
        for line in reversed([l.strip() for l in proc.stdout.splitlines() if l.strip()]):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if not payload:
            raise RuntimeError(f"Nao consegui ler o tempo do subprocesso.\n{proc.stdout}\n{proc.stderr}")
        times.append(float(payload["total_s"]))

    return {
        "version": "numba_cold",
        "mean_total_s": mean(times),
        "std_total_s": stdev(times) if len(times) > 1 else 0.0,
        "min_total_s": min(times),
        "max_total_s": max(times),
        "load_data_s": payload.get("load_data_s", 0.0),
        "reconstruct_rgb_s": payload.get("reconstruct_rgb_s", 0.0),
        "preprocess_s": payload.get("preprocess_s", 0.0),
        "grayscale_s": 0.0,
        "edges_s": 0.0,
        "n_images": int(payload.get("n_images", 0)),
        "repeat": repeat,
    }


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.child_numba_run:
        return run_numba_child(args.dataset, args.split, args.limit)

    from common.config import CIFAR10_DIR, CIFAR100_DIR

    data_dir = CIFAR10_DIR if args.dataset == "cifar10" else CIFAR100_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Base nao encontrada em: {data_dir}")

    from experiments.baseline.pipeline import process_pipeline as run_baseline
    from experiments.vectorized.pipeline import process_pipeline as run_vectorized
    from experiments.numba.pipeline import process_pipeline as run_numba
    from experiments.multiprocessing.pipeline import process_pipeline as run_mp

    warmup = not args.no_warmup
    common = dict(dataset=args.dataset, split=args.split, limit=args.limit)

    print(f"Raiz do projeto: {PROJECT_ROOT}")
    print(f"limit={args.limit} repeat={args.repeat} warmup={warmup} cold_numba={args.cold_numba}\n")

    rows: list[dict[str, Any]] = []
    rows.append(benchmark_one("baseline", run_baseline, repeat=args.repeat, **common))
    rows.append(benchmark_one("vectorized", run_vectorized, repeat=args.repeat, **common))
    rows.append(benchmark_one("numba", run_numba, repeat=args.repeat, warmup=warmup, **common))
    rows.append(
        benchmark_one("multiprocessing", run_mp, repeat=args.repeat, processes=args.processes, **common)
    )

    try:
        from experiments.gpu.pipeline import process_pipeline as run_gpu

        rows.append(
            benchmark_one("gpu", run_gpu, repeat=args.repeat, warmup=warmup, device=args.device, **common)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"GPU version skipped: {exc}")

    if args.cold_numba:
        rows.append(benchmark_numba_cold(args.dataset, args.split, args.limit, args.repeat))

    baseline_time = next((r["mean_total_s"] for r in rows if r["version"] == "baseline"), None)
    for r in rows:
        r["speedup_vs_baseline"] = (baseline_time / r["mean_total_s"]) if baseline_time else 0.0

    n_effective = rows[0]["n_images"]
    if n_effective < args.limit:
        print(f"AVISO: o split '{args.split}' tem apenas {n_effective} imagens (limit={args.limit}).\n")

    print("=== Comparison ===")
    for r in sorted(rows, key=lambda x: x["mean_total_s"]):
        print(
            f"{r['version']:<16} mean={r['mean_total_s']:>10.6f}s "
            f"std={r['std_total_s']:>9.6f}s speedup={r['speedup_vs_baseline']:>8.2f}x n={r['n_images']}"
        )

    csv_path = resolve_csv_path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV salvo em: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())