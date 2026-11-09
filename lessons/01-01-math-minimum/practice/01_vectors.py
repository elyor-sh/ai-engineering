"""Урок 1.1 · Эксперимент 1 — фрукты как векторы: расстояние и «карта смыслов».

Запуск из корня репозитория:
    uv run python lessons/01-01-math-minimum/practice/01_vectors.py
"""

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем сразу в файл, без окна
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

FEATURES = ["сладость", "кислотность", "хруст"]
FRUITS = {
    "яблоко": [6, 5, 8],
    "груша": [7, 3, 5],
    "лимон": [1, 10, 2],
    "лайм": [2, 9, 2],
    "банан": [8, 1, 1],
    # Задание 1.2: добавь сюда свой фрукт, например "клубника": [...]
}

# ── 1. Расстояние «по линейке» на чистом Python ──────────────────────────────
print("── 1. Евклидово расстояние руками ──")


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Разница по каждому признаку → в квадрат → сумма → корень (Пифагор для N измерений)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


print(f"лимон ↔ лайм:  {euclidean(FRUITS['лимон'], FRUITS['лайм']):.2f}")
print(f"лимон ↔ банан: {euclidean(FRUITS['лимон'], FRUITS['банан']):.2f}")

# ── 2. То же в numpy: фрукты складываем в матрицу ────────────────────────────
print("\n── 2. Матрица фруктов в numpy ──")
names = list(FRUITS)
X = np.array(list(FRUITS.values()), dtype=float)  # строки — фрукты, столбцы — признаки
print(f"Форма матрицы: {X.shape}  (фруктов × признаков)")
print(f"Вектор яблока: {X[0]},  размерность {X.shape[1]}")

# np.linalg.norm(a - b) — то же самое евклидово расстояние, только быстрее
print("\nТаблица попарных расстояний (меньше = похожее):")
print(" " * 8 + "".join(f"{n:>8}" for n in names))
for i, a in enumerate(names):
    row = "".join(f"{np.linalg.norm(X[i] - X[j]):8.1f}" for j in range(len(names)))
    print(f"{a:>8}{row}")

# ── 3. Ближайший сосед для каждого фрукта ────────────────────────────────────
print("\n── 3. Кто на кого похож ──")
for i, name in enumerate(names):
    distances = np.linalg.norm(X - X[i], axis=1)  # расстояние от X[i] до всех строк сразу
    distances[i] = np.inf  # себя не считаем
    j = int(distances.argmin())
    print(f"{name:>8} → ближе всего: {names[j]} ({distances[j]:.2f})")

# ── 4. Карта фруктов: берём 2 признака из 3 и рисуем точки ──────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], s=120)
for i, name in enumerate(names):
    ax.annotate(name, (X[i, 0], X[i, 1]), textcoords="offset points", xytext=(8, 6), fontsize=12)
ax.set_xlabel(FEATURES[0])
ax.set_ylabel(FEATURES[1])
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 11)
ax.set_title("Карта фруктов: похожие — рядом")
ax.grid(alpha=0.3)
path = OUT / "01_fruits_map.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
print(f"\n🖼  Карта сохранена: {path}\n   Открыть: open {path}")
