"""Урок 1.7 · Выбор токена одного шага: temperature, top-p и greedy.

Запуск:
    uv run python lessons/01-07-text-generation/practice/01_sampling.py

Распределение придумано вручную. Это НЕ текст, сгенерированный настоящей моделью.
"""

from pathlib import Path

import numpy as np


def token_distribution(logits: np.ndarray, *, temperature: float = 1.0, top_p: float = 1.0) -> np.ndarray:
    """Сначала temperature, затем top-p. Это порядок этой лаборатории, не всех API.

    При temperature=0 выбираем первый максимальный logit (явная конвенция ties).
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or logits.size == 0 or not np.isfinite(logits).all():
        raise ValueError("Нужен непустой вектор конечных logits")
    if not np.isfinite(temperature) or temperature < 0 or not np.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("temperature должна быть неотрицательной, top_p — в (0, 1]")
    if temperature == 0:
        probabilities = np.zeros_like(logits)
        probabilities[np.argmax(logits)] = 1.0
        return probabilities

    scaled = (logits - logits.max()) / temperature
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    if top_p == 1.0:
        return probabilities  # без фильтрации, даже если хвост меньше точности суммы
    order = np.argsort(-probabilities, kind="stable")
    cumulative = np.cumsum(probabilities[order])
    # Допускаем один ULP округления: вычисленное 0.5999999999999999 считаем достигшим 0.6.
    threshold = np.nextafter(top_p, 0.0)
    cutoff = min(np.searchsorted(cumulative, threshold, side="left") + 1, len(order))
    kept = order[:cutoff]  # включает токен, который первым достиг или пересёк порог
    filtered = np.zeros_like(probabilities)
    filtered[kept] = probabilities[kept]
    return filtered / filtered.sum()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = ["Париж", "Лион", "город", "бегемот"]
    logits = np.log([0.60, 0.25, 0.10, 0.05])
    scenarios = [
        ("Исходное распределение", 1.0, 1.0),
        ("Низкая температура", 0.5, 1.0),
        ("Ограничение top-p", 1.0, 0.8),
        ("Жадный выбор", 0.0, 1.0),
    ]
    print("Условный префикс: «Столица Франции — …»; logits заданы вручную.")
    print("Сэмплируем одну и ту же позицию, а не строим целые предложения.\n")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    for ax, (title, temperature, top_p) in zip(axes.flat, scenarios, strict=True):
        probabilities = token_distribution(logits, temperature=temperature, top_p=top_p)
        rng = np.random.default_rng(42)  # воспроизводимость этой программы, не гарантия API
        counts = np.bincount(rng.choice(len(tokens), size=1000, p=probabilities), minlength=len(tokens))
        print(f"{title}: T={temperature}, top_p={top_p}")
        for token, probability, count in zip(tokens, probabilities, counts, strict=True):
            print(f"  {token:8}: p={probability:.3f}, выбран {count} раз из 1000")
        ax.bar(tokens, probabilities, color="tab:blue")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title}\nT={temperature}, p={top_p}")
        ax.set_ylabel("Вероятность выбора токена")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = Path(__file__).parent / "out" / "01_sampling.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик: {path}")
    print("Вероятность продолжения не является вероятностью истины. Настоящая модель меняет logits после каждого токена.")


if __name__ == "__main__":
    main()
