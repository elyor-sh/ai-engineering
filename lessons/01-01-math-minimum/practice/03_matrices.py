"""Урок 1.1 · Эксперимент 3 — матрицы: машина преобразований и поиск одним умножением.

Запуск из корня репозитория:
    uv run python lessons/01-01-math-minimum/practice/03_matrices.py
"""

import time

import numpy as np

# ── 1. Матрица как машина преобразований: «вкус фрукта» → «пригодность для блюд» ─
print("── 1. Матрица превращает один вектор в другой ──")
FRUITS = {"яблоко": [6, 5, 8], "лимон": [1, 10, 2], "банан": [8, 1, 1]}
DISHES = ["смузи", "салат", "пирог"]

# Строки — блюда, столбцы — веса признаков [сладость, кислотность, хруст]
W = np.array(
    [
        [0.8, -0.3, 0.0],  # смузи: любит сладость, не любит кислоту, хруст всё равно
        [0.1, 0.3, 0.8],  # салат: хруст и немного кислинки
        [0.6, -0.1, 0.5],  # пирог: сладость и хруст
    ]
)
print(f"Матрица весов W: {W.shape} (блюда × признаки)\n")

for name, features in FRUITS.items():
    x = np.array(features, dtype=float)
    scores = W @ x  # = три скалярных произведения: x с каждой строкой W
    best = DISHES[int(scores.argmax())]
    pretty = ", ".join(f"{d} {s:.1f}" for d, s in zip(DISHES, scores, strict=True))
    print(f"  {name:<7} {features} → [{pretty}]  → лучше всего: {best}")

print("\nВсе фрукты сразу — одно умножение матрицы на матрицу:")
X = np.array(list(FRUITS.values()), dtype=float)  # 3 фрукта × 3 признака
print((W @ X.T).round(1), "\n← столбцы — фрукты, строки — блюда")
print("💡 Слой нейросети — это ровно такое умножение на матрицу весов (урок 1.2).")

# ── 2. Поиск по смыслу = умножение матрицы документов на вектор запроса ─────
print("\n── 2. Поиск по 100 000 «документов» ──")
rng = np.random.default_rng(42)
N, DIM = 100_000, 384  # 100 тыс. векторов по 384 числа — типичный размер небольших эмбеддингов

docs = rng.normal(size=(N, DIM)).astype(np.float32)
docs /= np.linalg.norm(docs, axis=1, keepdims=True)  # нормализуем: dot == cosine

# «Вопрос» похож на документ №777: берём его и добавляем шум
query = docs[777] + 0.05 * rng.normal(size=DIM).astype(np.float32)
query /= np.linalg.norm(query)
print(f"Матрица документов: {docs.shape}, {docs.nbytes / 1e6:.0f} МБ")

# а) Обычный цикл Python — только по первым 2 000 документам, иначе слишком долго
SAMPLE = 2_000
docs_list, query_list = docs[:SAMPLE].tolist(), query.tolist()
t0 = time.perf_counter()
_ = [sum(d * q for d, q in zip(doc, query_list, strict=True)) for doc in docs_list]
python_time = (time.perf_counter() - t0) * (N / SAMPLE)  # экстраполируем на все N

# б) numpy: одно умножение матрицы на вектор
t0 = time.perf_counter()
scores = docs @ query
numpy_time = time.perf_counter() - t0

top5 = np.argsort(-scores)[:5]
print(f"\n  Цикл Python (оценка на {N:,}): {python_time:7.2f} с")
print(f"  numpy, docs @ query:          {numpy_time:7.4f} с")
print(f"  Ускорение: ~{python_time / numpy_time:,.0f}×")
print(f"\n  Топ-5 документов: {top5.tolist()}")
print(f"  Их сходство:      {[round(float(s), 3) for s in scores[top5]]}")
print("  ✅ Документ №777 на первом месте" if top5[0] == 777 else "  ❌ Что-то пошло не так")
print(
    "\n💡 numpy делает то же, что видеокарта, но на процессоре: много одинаковых\n"
    "   умножений сразу (векторизация). GPU ускоряет это ещё в десятки раз."
)
