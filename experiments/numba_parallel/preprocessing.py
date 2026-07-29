"""Numba implementation for grayscale + Sobel, parallelized over images.

Diferenca em relacao a experiments/numba:
- o laco externo (sobre imagens) usa `prange` e a funcao e compilada com
  `parallel=True`, entao o Numba distribui as imagens entre threads;
- a aritmetica por pixel e **identica** a da versao serial, propositalmente.
  Cada imagem e processada de forma independente e cada thread escreve apenas
  na sua propria fatia dos arrays de saida, entao a saida e bit a bit igual a
  da versao serial (`benchmark/verify_pipeline.py` checa isso com atol=0).

Se precisar desligar o cache do JIT (versoes antigas do Numba tinham
limitacoes com `cache=True` junto de `parallel=True`), exporte
NUMBA_PARALLEL_CACHE=0 antes de rodar.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

try:
    from numba import config as numba_config
    from numba import get_num_threads, njit, prange, set_num_threads
except Exception as exc:  # pragma: no cover
    njit = None
    prange = range
    numba_config = None
    get_num_threads = None
    set_num_threads = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None


_CACHE = os.environ.get("NUMBA_PARALLEL_CACHE", "1") != "0"


def _require_numba() -> None:
    if njit is None:
        raise RuntimeError(
            "Numba is required for the numba_parallel version."
        ) from _NUMBA_IMPORT_ERROR


def max_threads() -> int:
    """Numero maximo de threads que o Numba pode usar neste processo."""
    _require_numba()
    return int(numba_config.NUMBA_NUM_THREADS)


def resolve_threads(threads: Optional[int]) -> int:
    """Aplica o numero de threads pedido, limitado ao maximo disponivel."""
    _require_numba()
    hard_max = max_threads()

    if threads is None:
        return int(get_num_threads())

    threads = max(1, int(threads))
    if threads > hard_max:
        print(
            f"AVISO: {threads} threads pedidas, mas o Numba permite no maximo "
            f"{hard_max} neste processo. Use NUMBA_NUM_THREADS para elevar o teto."
        )
        threads = hard_max

    set_num_threads(threads)
    return threads


if njit is not None:

    @njit(cache=True)
    def _grayscale_into(image_rgb, gray):
        for i in range(32):
            for j in range(32):
                r = float(image_rgb[i, j, 0])
                g = float(image_rgb[i, j, 1])
                b = float(image_rgb[i, j, 2])
                gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b

    @njit(cache=True)
    def _sobel_into(gray_image, edges):
        for i in range(32):
            for j in range(32):
                gx = 0.0
                gy = 0.0

                for di in (-1, 0, 1):
                    ii = i + di
                    if ii < 0:
                        ii = 0
                    elif ii > 31:
                        ii = 31

                    for dj in (-1, 0, 1):
                        jj = j + dj
                        if jj < 0:
                            jj = 0
                        elif jj > 31:
                            jj = 31

                        val = float(gray_image[ii, jj])

                        if di == -1 and dj == -1:
                            gx += -1.0 * val
                            gy += -1.0 * val
                        elif di == -1 and dj == 0:
                            gy += -2.0 * val
                        elif di == -1 and dj == 1:
                            gx += 1.0 * val
                            gy += -1.0 * val
                        elif di == 0 and dj == -1:
                            gx += -2.0 * val
                        elif di == 0 and dj == 1:
                            gx += 2.0 * val
                        elif di == 1 and dj == -1:
                            gx += -1.0 * val
                            gy += 1.0 * val
                        elif di == 1 and dj == 0:
                            gy += 2.0 * val
                        elif di == 1 and dj == 1:
                            gx += 1.0 * val
                            gy += 1.0 * val

                edges[i, j] = (gx * gx + gy * gy) ** 0.5

        max_val = 0.0
        for i in range(32):
            for j in range(32):
                if edges[i, j] > max_val:
                    max_val = edges[i, j]

        if max_val > 0.0:
            for i in range(32):
                for j in range(32):
                    edges[i, j] = edges[i, j] / max_val

    @njit(parallel=True, cache=_CACHE)
    def process_images_numba_parallel(images_rgb):
        n = images_rgb.shape[0]
        grays = np.empty((n, 32, 32), dtype=np.float32)
        edges = np.empty((n, 32, 32), dtype=np.float32)

        # cada iteracao escreve apenas em grays[idx] e edges[idx]:
        # nenhuma escrita compartilhada, nenhuma reducao entre threads
        for idx in prange(n):
            _grayscale_into(images_rgb[idx], grays[idx])
            _sobel_into(grays[idx], edges[idx])

        return grays, edges


def process_images(
    images_rgb: np.ndarray,
    threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    _require_numba()

    if images_rgb.ndim != 4 or images_rgb.shape[-1] != 3:
        raise ValueError("images_rgb must have shape (N, 32, 32, 3).")

    resolve_threads(threads)
    return process_images_numba_parallel(np.ascontiguousarray(images_rgb))


def warmup(threads: Optional[int] = None) -> int:
    """Compila o JIT (e aquece o pool de threads) com um lote minimo.

    Retorna o numero de threads em uso. Chame antes de medir tempo, para nao
    contabilizar compilacao dentro da medicao.
    """
    _require_numba()
    used = resolve_threads(threads)
    dummy = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    process_images_numba_parallel(dummy)
    return used
