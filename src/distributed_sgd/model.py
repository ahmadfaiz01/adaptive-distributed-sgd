from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def make_dataset(n_samples: int, n_features: int, n_classes: int, seed: int):
    informative = max(4, min(n_features, n_features // 2))
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=informative,
        n_redundant=max(0, min(n_features - informative, n_features // 4)),
        n_classes=n_classes,
        class_sep=1.5,
        random_state=seed,
    )
    x = x.astype(np.float64)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    y = y.astype(np.int64)
    return train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)


def init_params(n_features: int, n_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=(n_features, n_classes))


def softmax_loss_and_grad(params: np.ndarray, x: np.ndarray, y: np.ndarray):
    logits = x @ params
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(y.size), y] + 1e-12).mean()
    probs[np.arange(y.size), y] -= 1.0
    grad = x.T @ probs / y.size
    return float(loss), grad


def evaluate_loss(params: np.ndarray, x: np.ndarray, y: np.ndarray):
    logits = x @ params
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return float(-np.log(probs[np.arange(y.size), y] + 1e-12).mean())


def shard_dataset(x: np.ndarray, y: np.ndarray, workers: int):
    x_parts = np.array_split(x, workers)
    y_parts = np.array_split(y, workers)
    return list(zip(x_parts, y_parts))
