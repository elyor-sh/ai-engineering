"""Урок 1.5 · Причинная маска и проверка, что будущее не влияет на прошлое.

Запуск:
    uv run python lessons/01-05-transformer-attention/practice/02_causal_mask.py

Q = K = V = X: для наглядности проекции единичные, все числа искусственные.
Проверка перестановок относится к полному attention БЕЗ маски и позиционных сигналов.
"""

from pathlib import Path

import numpy as np
from attention_core import attention


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=3, suppress=True)
    labels = ["Я", "вижу", "новый", "дом"]
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    full_output, full_weights = attention(x, x, x)
    causal_output, causal_weights = attention(x, x, x, causal=True)
    print("Без маски:\n", full_weights)
    print("\nС причинной маской (справа от диагонали нули):\n", causal_weights)
    print("Суммы строк:", causal_weights.sum(axis=1))

    changed = x.copy()
    changed[-1] = [4.0, -3.0]
    new_full, _ = attention(changed, changed, changed)
    new_causal, _ = attention(changed, changed, changed, causal=True)
    print("\nИзменили только последний входной вектор:")
    print("  Ранние результаты без маски сохранились:", np.allclose(full_output[:-1], new_full[:-1]))
    print("  Ранние результаты с маской сохранились:", np.allclose(causal_output[:-1], new_causal[:-1]))
    print("  Вторая проверка должна дать True: ранним позициям нельзя видеть изменённое будущее.")

    order = [2, 0, 3, 1]
    permuted_output, _ = attention(x[order], x[order], x[order])
    print("\nПереставили входы полного attention без маски и позиционных сигналов:")
    print("  Выходы переставились так же:", np.allclose(permuted_output, full_output[order]))
    print("  Выходная последовательность осталась буквально прежней:", np.allclose(permuted_output, full_output))
    print("  Это разные утверждения. Позиционное кодирование в этом опыте отсутствует.\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, weights, title in zip(axes, [full_weights, causal_weights], ["Без маски", "Причинная маска"], strict=True):
        ax.imshow(weights, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Источники")
        ax.set_ylabel("Получатели")
        ax.set_title(title)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", color="white" if weights[i, j] > 0.5 else "black")
    fig.tight_layout()
    path = Path(__file__).parent / "out" / "02_causal_mask.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Сравнение сохранено: {path}")


if __name__ == "__main__":
    main()
