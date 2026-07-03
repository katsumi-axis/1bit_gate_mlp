from __future__ import annotations

import pickle
import urllib.request
from pathlib import Path

import numpy as np

MNIST1D_URL = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
MNIST1D_FILE = "mnist1d_data.pkl"


def download_mnist1d(data_dir: Path) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    target = raw_dir / MNIST1D_FILE
    if not target.exists():
        print(f"Downloading {MNIST1D_URL}")
        urllib.request.urlretrieve(MNIST1D_URL, target)
    return target


def _normalization_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return mean, np.maximum(std, 1e-6)


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) - mean) / std


def load_mnist1d(data_dir: Path) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    dataset_path = download_mnist1d(data_dir)
    with dataset_path.open("rb") as f:
        data = pickle.load(f)

    mean, std = _normalization_stats(data["x"])
    train_x = _normalize(data["x"], mean, std)
    train_y = data["y"].astype(np.int32)
    test_x = _normalize(data["x_test"], mean, std)
    test_y = data["y_test"].astype(np.int32)
    return (train_x, train_y), (test_x, test_y)


def batch_iter(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator,
):
    indices = np.arange(x.shape[0])
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, x.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], y[batch_indices]
