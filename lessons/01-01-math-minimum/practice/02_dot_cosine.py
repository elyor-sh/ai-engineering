"""Урок 1.1 · Эксперимент 2 — скалярное произведение и косинусное сходство.

Запуск из корня репозитория:
    uv run python lessons/01-01-math-minimum/practice/02_dot_cosine.py
"""

from collections.abc import Sequence

import numpy as np

# ── 1. Скалярное произведение = «мера согласия» ─────────────────────────────
print("── 1. Рекомендации фильмов через скалярное произведение ──")
print("Признаки: [боевик, комедия, мелодрама]\n")


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Перемножаем попарно и складываем."""
    return sum(x * y for x, y in zip(a, b, strict=True))


alex = [5, 2, -3]  # обожает боевики, нормально к комедиям, терпеть не может мелодрамы
MOVIES = {
    "Боевик": [9, 1, 0],
    "Комедийный боевик": [6, 7, 0],
    "Романтическая комедия": [0, 6, 8],
    "Мелодрама": [0, 1, 9],
    "Документалка": [0, 0, 0],
}

print(f"Вкус Алекса: {alex}")
for title, movie in sorted(MOVIES.items(), key=lambda kv: -dot(alex, kv[1])):
    score = dot(alex, movie)
    verdict = "👍 рекомендуем" if score > 0 else ("👎 мимо" if score < 0 else "🤷 без разницы")
    print(f"  {title:<22} {movie!s:<12} → {score:>4}  {verdict}")

a, b = np.array(alex), np.array(MOVIES["Боевик"])
print(f"\nТо же в numpy: np.dot = {np.dot(a, b)},  оператор @ = {a @ b}")

# ── 2. Длина вектора и нормализация ─────────────────────────────────────────
print("\n── 2. Длина и нормализация ──")
v = np.array([3.0, 4.0])
length = np.linalg.norm(v)  # корень из (3² + 4²) = 5
unit = v / length
print(f"v = {v}, длина = {length}")
print(f"нормализованный v = {unit}, длина = {np.linalg.norm(unit):.3f}  ← направление то же, длина 1")

# ── 3. Косинус vs расстояние на «текстах» ───────────────────────────────────
print("\n── 3. Почему для текстов косинус лучше расстояния ──")
VOCAB = ["кошка", "спит", "налог"]
TEXTS = {
    "кошка спит": [1, 1, 0],
    "кошка спит ×3 (повтор)": [3, 3, 0],
    "налог": [0, 0, 1],
    "кошка платит налог": [1, 0, 1],
}
print(f"Тексты описаны счётчиками слов {VOCAB}\n")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


base_name = "кошка спит"
base = np.array(TEXTS[base_name], dtype=float)
print(f"Сравниваем с «{base_name}»:")
print(f"  {'текст':<24}{'расстояние':>12}{'косинус':>10}")
for name, vec in TEXTS.items():
    if name == base_name:
        continue
    vec = np.array(vec, dtype=float)
    print(f"  {name:<24}{np.linalg.norm(base - vec):>12.2f}{cosine(base, vec):>10.2f}")
print(
    "\n  ⚠️  По расстоянию «кошка спит ×3» ДАЛЬШЕ, чем «налог»!\n"
    "     По косинусу — идентичен (1.00): направление то же, просто стрелка длиннее."
)

# ── 4. После нормализации dot == cosine ─────────────────────────────────────
print("\n── 4. Нормализуем → скалярное произведение = косинус ──")
x, y = np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 4.0])
x_unit, y_unit = x / np.linalg.norm(x), y / np.linalg.norm(y)
print(f"cosine(x, y)            = {cosine(x, y):.4f}")
print(f"dot(x_unit, y_unit)     = {x_unit @ y_unit:.4f}  ← то же самое")
print(f"dot(x, y) без нормализации = {x @ y:.4f}  ← зависит от длины")
