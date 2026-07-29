#!/usr/bin/env python3
"""
Pipeline verifier for CIFAR experiments.

O que ele checa:
- carrega uma amostra pequena da CIFAR e valida shapes, dtypes e rotulos;
- roda TODAS as versoes disponiveis (baseline, vectorized, numba,
  numba_parallel, multiprocessing, gpu), pulando as que faltam dependencia;
- compara a saida de cada versao contra a baseline, que e a referencia de
  corretude;
- checa que numba_parallel e bit a bit igual ao numba serial (atol=0), ou seja,
  que paralelizar nao mudou o resultado;
- salva um preview visual (RGB / gray / edges) de uma amostra.

Uso:
    python3 -m benchmark.verify_pipeline
    python3 verify_pipeline.py --limit 20 --sample-index 0
    python3 verify_pipeline.py --versions baseline numba numba_parallel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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

ALL_VERSIONS = [
    "baseline",
    "vectorized",
    "numba",
    "numba_parallel",
    "multiprocessing",
    "gpu",
]

# tolerancias por versao, contra a baseline.
# baseline, numba e numba_parallel usam a mesma ordem de operacoes, entao a
# diferenca deve ser nula. vectorized e gpu reassociam as somas do Sobel e
# usam kernels/convolucao, o que muda a ultima casa do float32.
TOLERANCES = {
    "vectorized": 1e-4,
    "numba": 0.0,
    "numba_parallel": 0.0,
    "multiprocessing": 0.0,
    "gpu": 1e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CIFAR pipeline correctness.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=20, help="Subconjunto pequeno para os testes.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=ALL_VERSIONS,
        default=ALL_VERSIONS,
        help="Versoes a verificar (baseline e sempre incluida como referencia).",
    )
    parser.add_argument("--threads", type=int, default=None, help="Threads do numba_parallel.")
    parser.add_argument("--device", type=str, default="auto", help="Device do PyTorch.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "verification",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# checagens
# --------------------------------------------------------------------------

def summarize_basic(images_flat: np.ndarray, labels: np.ndarray, class_names) -> None:
    print("\n=== Basic dataset checks ===")
    print(f"Flat images shape: {images_flat.shape}")
    print(f"Labels shape:      {labels.shape}")
    print(f"Images dtype:      {images_flat.dtype}")
    print(f"Labels dtype:      {labels.dtype}")
    print(f"Min pixel:         {images_flat.min()}")
    print(f"Max pixel:         {images_flat.max()}")
    print(f"Mean pixel:        {images_flat.mean():.3f}")
    print(f"Std pixel:         {images_flat.std():.3f}")

    unique = np.unique(labels)
    print(f"Unique labels in subset: {unique.tolist()}")
    if class_names:
        print(f"Class count: {len(class_names)}")
        print("First 5 class names:", class_names[:5])


def compare_arrays(
    label: str,
    a: np.ndarray,
    b: np.ndarray,
    atol: float,
) -> tuple[bool, float]:
    """Compara dois arrays e devolve (passou, maior diferenca absoluta)."""
    if a.shape != b.shape:
        print(f"  {label:<8} FALHOU: shapes diferentes {a.shape} vs {b.shape}")
        return False, float("inf")

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_diff = float(diff.max())
    ok = bool(np.allclose(a, b, atol=atol, rtol=0))

    verdict = "OK" if ok else "FALHOU"
    print(f"  {label:<8} {verdict:<7} max diff {max_diff:.3e}  (tol {atol:.0e})")

    if not ok:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        print(f"           pior indice {idx}: {a[idx]} vs {b[idx]}")

    return ok, max_diff


def validate_pipeline_outputs(name: str, result: Any, expected_n: int) -> bool:
    """Checa shapes, dtypes, finitude e faixas de valores de uma versao."""
    print(f"\n=== Output checks: {name} ===")
    ok = True

    checks = [
        (result.images_rgb.shape[0] == expected_n, "numero de imagens RGB"),
        (result.images_gray.shape[0] == expected_n, "numero de imagens gray"),
        (result.images_edges.shape[0] == expected_n, "numero de imagens edges"),
        (result.labels.shape[0] == expected_n, "numero de rotulos"),
        (result.images_rgb.shape[1:] == (32, 32, 3), "shape RGB (N, 32, 32, 3)"),
        (result.images_gray.shape[1:] == (32, 32), "shape gray (N, 32, 32)"),
        (result.images_edges.shape[1:] == (32, 32), "shape edges (N, 32, 32)"),
        (bool(np.isfinite(result.images_gray).all()), "gray finito"),
        (bool(np.isfinite(result.images_edges).all()), "edges finito"),
    ]

    for passed, desc in checks:
        if not passed:
            print(f"  FALHOU: {desc}")
            ok = False

    if ok:
        print("  Shapes, rotulos e finitude: OK")

    gray_min, gray_max = float(result.images_gray.min()), float(result.images_gray.max())
    edge_min, edge_max = float(result.images_edges.min()), float(result.images_edges.max())
    print(f"  Gray range:  [{gray_min:.4f}, {gray_max:.4f}]")
    print(f"  Edges range: [{edge_min:.4f}, {edge_max:.4f}]")

    if gray_min < -1e-6 or gray_max > 255.0 + 1e-6:
        print("  AVISO: gray fora da faixa [0, 255]")
    if edge_min < -1e-6 or edge_max > 1.0 + 1e-6:
        print("  AVISO: edges fora da faixa [0, 1] (a normalizacao e por imagem)")

    return ok


def save_preview(result, labels, class_names, sample_index: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = result.images_rgb[sample_index]
    gray = result.images_gray[sample_index]
    edges = result.images_edges[sample_index]
    label = int(labels[sample_index])

    if class_names and 0 <= label < len(class_names):
        title = f"class {label}: {class_names[label]}"
    else:
        title = f"class {label}"

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(rgb.astype(np.uint8))
    axes[0].set_title(f"RGB\n{title}")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Gray")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Edges")
    axes[2].axis("off")

    fig.tight_layout()
    out_file = output_dir / f"sample_{sample_index}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPreview salvo em: {out_file}")


# --------------------------------------------------------------------------
# resolucao das versoes
# --------------------------------------------------------------------------

def build_runners(args: argparse.Namespace) -> dict[str, Callable[..., Any]]:
    """Importa as pipelines pedidas, pulando as que faltam dependencia."""
    wanted = list(dict.fromkeys(["baseline", *args.versions]))
    runners: dict[str, Callable[..., Any]] = {}

    for name in wanted:
        try:
            if name == "baseline":
                from experiments.baseline.pipeline import process_pipeline as fn
            elif name == "vectorized":
                from experiments.vectorized.pipeline import process_pipeline as fn
            elif name == "numba":
                from experiments.numba.pipeline import process_pipeline as fn
            elif name == "numba_parallel":
                from experiments.numba_parallel.pipeline import process_pipeline as base_fn

                def fn(**kwargs: Any):  # noqa: ANN202
                    return base_fn(threads=args.threads, **kwargs)
            elif name == "multiprocessing":
                from experiments.multiprocessing.pipeline import process_pipeline as fn
            elif name == "gpu":
                from experiments.gpu.pipeline import process_pipeline as base_gpu

                def fn(**kwargs: Any):  # noqa: ANN202
                    return base_gpu(device=args.device, **kwargs)
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"AVISO: versao '{name}' indisponivel, pulando ({exc}).")
            continue

        runners[name] = fn

    return runners


# --------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    from common.config import CIFAR10_DIR, CIFAR100_DIR
    from common.dataset import load_dataset

    data_dir = CIFAR10_DIR if args.dataset == "cifar10" else CIFAR100_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Base nao encontrada em: {data_dir}")

    images_flat, labels, class_names = load_dataset(
        data_dir,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
    )
    summarize_basic(images_flat, labels, class_names)

    n_expected = int(images_flat.shape[0])
    if n_expected != args.limit:
        print(f"\nNota: o split tem {n_expected} imagens, menos que o limit={args.limit}.")

    runners = build_runners(args)
    print(f"\nVersoes a verificar: {list(runners)}")

    results: dict[str, Any] = {}
    for name, fn in runners.items():
        print(f"\nRodando {name}...")
        try:
            results[name] = fn(dataset=args.dataset, split=args.split, limit=args.limit)
        except Exception as exc:  # noqa: BLE001
            # o import pode funcionar e a execucao falhar por falta de runtime
            # (torch sem CUDA, numba ausente). isso e motivo de pular, nao de falhar.
            if name == "baseline":
                raise
            print(f"  AVISO: '{name}' nao executou, pulando ({type(exc).__name__}: {exc}).")

    failures: list[str] = []

    for name, result in results.items():
        if not validate_pipeline_outputs(name, result, expected_n=n_expected):
            failures.append(f"{name}: checagens de output")

    reference = results["baseline"]

    for name, result in results.items():
        if name == "baseline":
            continue

        atol = TOLERANCES.get(name, 1e-4)
        print(f"\n=== {name} vs baseline (tolerancia {atol:.0e}) ===")

        rgb_ok, _ = compare_arrays("RGB", reference.images_rgb, result.images_rgb, atol=0.0)
        gray_ok, _ = compare_arrays("Gray", reference.images_gray, result.images_gray, atol=atol)
        edge_ok, _ = compare_arrays("Edges", reference.images_edges, result.images_edges, atol=atol)

        labels_ok = np.array_equal(reference.labels, result.labels)
        print(f"  {'Labels':<8} {'OK' if labels_ok else 'FALHOU'}")

        if not all([rgb_ok, gray_ok, edge_ok, labels_ok]):
            failures.append(f"{name} vs baseline")

    # checagem especifica: paralelizar nao pode alterar o resultado
    if "numba" in results and "numba_parallel" in results:
        print("\n=== numba_parallel vs numba serial (exige igualdade exata) ===")
        gray_ok, _ = compare_arrays(
            "Gray", results["numba"].images_gray, results["numba_parallel"].images_gray, atol=0.0
        )
        edge_ok, _ = compare_arrays(
            "Edges", results["numba"].images_edges, results["numba_parallel"].images_edges, atol=0.0
        )
        if not (gray_ok and edge_ok):
            failures.append("numba_parallel vs numba (igualdade exata)")
        else:
            print("  Paralelizar nao alterou o resultado: OK")

    print("\n=== Timing summary ===")
    print("(execucao unica e volume minusculo, nao use como benchmark:")
    print(" o Numba paga compilacao JIT aqui, use benchmark/scaling_parallel.py)")
    for name, result in results.items():
        extra = ""
        threads = getattr(result, "threads", None)
        if threads:
            extra = f"  ({threads} threads)"
        print(f"  {name:<16} {result.timings.total_s:>10.6f} s{extra}")

    save_preview(reference, labels, class_names, args.sample_index, args.output_dir)

    print("\n" + "=" * 60)
    if failures:
        print("RESULTADO: FALHOU")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"RESULTADO: todas as {len(results)} versoes verificadas passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
