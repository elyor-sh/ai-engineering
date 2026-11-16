"""Урок 1.5 · Одна голова attention: Q, K, V, веса и взвешенная сумма.

Запуск из корня проекта:
    uv run python lessons/01-05-transformer-attention/practice/01_attention_weights.py

Числа заданы вручную. Подписи слов — иллюстрация, а не результат обучения языку.
"""

from pathlib import Path

import numpy as np
from attention_core import attention


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=3, suppress=True)
    labels = ["кошка", "устала", "она"]
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    wq = np.array([[1.0, 0.2], [0.1, 1.0]])
    wk = np.array([[1.0, 0.0], [0.0, 0.7]])
    wv = np.array([[2.0, -1.0], [1.0, 3.0]])

    q, k, v = x @ wq, x @ wk, x @ wv
    print("Входные векторы X:\n", x)
    print("\nQ = X @ Wq:\n", q)
    print("\nK = X @ Wk:\n", k)
    print("\nV = X @ Wv:\n", v)
    output, weights = attention(q, k, v)
    print("\nВеса: строки — получатели, столбцы — источники:\n", weights)
    print("Суммы строк:", weights.sum(axis=1))
    print("\nОбновления позиций = веса @ V:\n", output)

    row = 2
    by_hand = np.sum(weights[row, :, None] * v, axis=0)
    print(f"\nДля позиции «{labels[row]}» сумма вкладов V: {by_hand}")
    print("Совпадает с матричным вычислением:", np.allclose(by_hand, output[row]))
    print("Это числовое смешивание. Модель здесь не обучалась разрешать местоимения.\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(weights, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Источники: чьи Key и Value используем")
    ax.set_ylabel("Получатели: чей Query")
    ax.set_title("Одна голова внимания без причинной маски")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", color="white" if weights[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, label="Вес (не вероятность истины)")
    fig.tight_layout()
    path = Path(__file__).parent / "out" / "01_attention_weights.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Карта сохранена: {path}")


if __name__ == "__main__":
    main()
