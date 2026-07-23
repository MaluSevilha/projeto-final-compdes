"""Leitura e preparação da base CIFAR em formato Python."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def unpickle(path: Path) -> Dict[str, Any]:
    """Carrega um arquivo pickle da CIFAR."""
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def labels_key(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return "labels"
    if dataset == "cifar100":
        return "fine_labels"
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def meta_file(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return "batches.meta"
    if dataset == "cifar100":
        return "meta"
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")


def batch_files(dataset: str, split: str) -> List[str]:
    dataset = dataset.lower()
    split = split.lower()

    if dataset == "cifar10":
        if split == "train":
            return [f"data_batch_{i}" for i in range(1, 6)]
        if split == "test":
            return ["test_batch"]
    elif dataset == "cifar100":
        if split == "train":
            return ["train"]
        if split == "test":
            return ["test"]

    raise ValueError("Invalid dataset/split combination.")


def load_class_names(data_dir: Path, dataset: str) -> Optional[List[str]]:
    """Carrega os nomes das classes, se existir arquivo de metadados."""
    path = data_dir / meta_file(dataset)
    if not path.exists():
        return None

    meta = unpickle(path)

    if dataset.lower() == "cifar10":
        names = meta.get("label_names")
    else:
        names = meta.get("fine_label_names")

    if not names:
        return None

    return [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]


def load_one_batch(data_dir: Path, dataset: str, batch_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carrega um batch em formato CIFAR Python."""
    path = data_dir / batch_name
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {path}")

    batch = unpickle(path)
    key = labels_key(dataset)

    images = np.asarray(batch["data"], dtype=np.uint8)
    labels = np.asarray(batch[key], dtype=np.int64)
    return images, labels


def load_dataset(
    data_dir: Path,
    dataset: str = "cifar10",
    split: str = "train",
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Carrega vários batches e devolve imagens achatadas, rótulos e nomes das classes."""
    images_list = []
    labels_list = []
    remaining = None if limit is None else int(limit)

    for batch_name in batch_files(dataset, split):
        if remaining is not None and remaining <= 0:
            break

        images, labels = load_one_batch(data_dir, dataset, batch_name)

        if remaining is not None:
            take = min(remaining, len(images))
            images = images[:take]
            labels = labels[:take]
            remaining -= take

        images_list.append(images)
        labels_list.append(labels)

    if not images_list:
        raise FileNotFoundError(
            f"Nenhum batch carregado em {data_dir}. Verifique o caminho e o dataset."
        )

    images_flat = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    class_names = load_class_names(data_dir, dataset)

    return images_flat, labels, class_names


def reconstruct_image(flat_image: np.ndarray) -> np.ndarray:
    """
    Reconstrói uma imagem CIFAR de (3072,) para (32, 32, 3).
    A base armazena os canais em blocos RGB.
    """
    if flat_image.shape != (3072,):
        raise ValueError(f"Expected shape (3072,), got {flat_image.shape}")
    return flat_image.reshape(3, 32, 32).transpose(1, 2, 0)


def reconstruct_images(images_flat: np.ndarray) -> np.ndarray:
    """Reconstrói um lote de imagens de (N, 3072) para (N, 32, 32, 3)."""
    if images_flat.ndim != 2 or images_flat.shape[1] != 3072:
        raise ValueError("images_flat must have shape (N, 3072).")
    n = images_flat.shape[0]
    return images_flat.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)
