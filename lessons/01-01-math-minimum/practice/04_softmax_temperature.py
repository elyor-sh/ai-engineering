"""Урок 1.1 · Эксперимент 4 — softmax и температура: как модель выбирает следующий токен.

Запуск из корня репозитория:
    uv run python lessons/01-01-math-minimum/practice/04_softmax_temperature.py
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

PROMPT = "Столица Франции — это …"
TOKENS = ["Париж", "Лион", "город", "прекрасный", "бегемот"]
LOGITS = [5.0, 2.0, 1.0, 0.5, -3.0]  # «сырые баллы» последнего слоя модели


def softmax(logits: list[float], temperature: float = 1.0) -> np.ndarray:
    """Сырые баллы → вероятности. Температура делит баллы ДО экспоненты."""
    z = np.asarray(logits, dtype=float)
    if temperature == 0:  # предел T → 0: всегда лидер (жадный выбор)
        probs = np.zeros_like(z)
        probs[z.argmax()] = 1.0
        return probs
    z = z / temperature
    z = z - z.max()  # трюк устойчивости: результат тот же, но exp() не переполнится
    e = np.exp(z)
    return e / e.sum()


# ── 1. Softmax по шагам ─────────────────────────────────────────────────────
print(f"Текст: «{PROMPT}»\n")
print("── 1. Softmax по шагам (T = 1) ──")
e = np.exp(LOGITS)
for tok, logit, ex in zip(TOKENS, LOGITS, e, strict=True):
    print(f"  {tok:<11} балл {logit:>5}  →  exp = {ex:>8.2f}  →  {ex / e.sum():6.1%}")
print(f"  {'':<11} сумма exp = {e.sum():.2f};  сумма вероятностей = {(e / e.sum()).sum():.1f}")

# ── 2. Как температура меняет распределение ─────────────────────────────────
TEMPS = [0, 0.5, 1.0, 2.0, 5.0]
print("\n── 2. Температура: острое ↔ плоское распределение ──")
print(f"  {'T':>5}" + "".join(f"{t:>12}" for t in TOKENS))
for t in TEMPS:
    probs = softmax(LOGITS, t)
    print(f"  {t:>5}" + "".join(f"{p:>12.1%}" for p in probs))

# ── 3. Сэмплирование: тянем 1000 «лотерейных билетов» ───────────────────────
print("\n── 3. 1000 выборов следующего токена ──")
rng = np.random.default_rng(0)
for t in [0.5, 1.0, 2.0]:
    picks = rng.choice(TOKENS, size=1000, p=softmax(LOGITS, t))
    counts = Counter(picks.tolist())
    pretty = ", ".join(f"{tok} {counts.get(tok, 0)}" for tok in TOKENS)
    print(f"  T = {t}: {pretty}")
print("  💡 При высокой T иногда выпадает даже «бегемот» — вот откуда бред на высокой температуре.")

# ── 4. График ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(TEMPS), figsize=(16, 3.6), sharey=True)
for ax, t in zip(axes, TEMPS, strict=True):
    probs = softmax(LOGITS, t)
    ax.bar(TOKENS, probs, color="tab:blue" if t <= 1 else "tab:orange")
    ax.set_title(f"T = {t}")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("вероятность")
fig.suptitle(f"«{PROMPT}» — распределение следующего токена при разной температуре", y=1.04)
path = OUT / "04_softmax_temperature.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
print(f"\n🖼  График сохранён: {path}\n   Открыть: open {path}")
