"""Математика одной головы scaled dot-product attention на NumPy.

Без обучения, dropout, нескольких голов, KV-cache, MLP и позиционного механизма.
Причинная маска здесь поддерживает только полное self-attention с одинаковой длиной Q и K.
"""

import numpy as np


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Устойчивый softmax по источникам (столбцам). -inf означает запрещённую связь."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or 0 in scores.shape:
        raise ValueError("Нужна непустая матрица баллов")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Баллы содержат NaN или положительную бесконечность")
    maximum = scores.max(axis=1, keepdims=True)
    if np.isneginf(maximum).any():
        raise ValueError("Нельзя замаскировать все источники в строке")
    exponents = np.exp(scores - maximum)  # вычитание максимума не меняет вероятности
    return exponents / exponents.sum(axis=1, keepdims=True)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (обновления позиций, веса внимания).

    Q: (получатели, размерность ключей).
    K: (источники, размерность ключей).
    V: (источники, размерность передаваемой информации).
    """
    q, k, v = (np.asarray(matrix, dtype=float) for matrix in (q, k, v))
    for matrix in (q, k, v):
        if matrix.ndim != 2 or 0 in matrix.shape or not np.isfinite(matrix).all():
            raise ValueError("Q, K и V должны быть непустыми матрицами конечных чисел")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Размерности Q, K и V не согласованы")
    if causal and q.shape[0] != k.shape[0]:
        raise ValueError("Учебная причинная маска ожидает одинаковую длину Q и K")

    scores = q @ k.T / np.sqrt(k.shape[1])
    if causal:
        future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)
    weights = softmax_rows(scores)
    output = weights @ v  # взвешенная сумма, а не случайный выбор одного V
    return output, weights
