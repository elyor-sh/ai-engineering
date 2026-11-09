"""Урок 1.2 · Обучаем один линейный нейрон: предсказание → ошибка → градиенты → шаг.

Запуск из корня проекта:
    uv run python lessons/01-02-neural-networks/practice/02_learning.py

Только искусственные данные. Это не оценщик реальных цен квартир.
"""

from pathlib import Path

import numpy as np


def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Прямой проход: один вход, один вес, одно смещение. Активация здесь не нужна."""
    return x * w + b


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Средний квадрат ошибки — ошибки разных знаков не уничтожают друг друга."""
    return float(np.mean((predictions - targets) ** 2))


def gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    """Производные MSE по весу и смещению для одного линейного нейрона.

    Выводить эти выражения для урока не обязательно. Главное — видеть связь
    «ошибка → чувствительность → обновление». Библиотеки делают это автоматически.
    """
    error = predict(x, w, b) - y
    dw = float(2 * np.mean(error * x))
    db = float(2 * np.mean(error))
    return dw, db


def train(x: np.ndarray, y: np.ndarray, *, lr: float = 0.1, steps: int = 400) -> tuple[float, float, list[float]]:
    """Полный батч: на каждом шаге считаем градиенты по всем учебным примерам.

    Возвращает вес, смещение и историю потерь (начальное значение + каждый шаг).
    Здесь нулевая инициализация подходит: нейрон один, проблемы симметрии нет.
    """
    w, b = 0.0, 0.0
    losses = [mse(predict(x, w, b), y)]
    for _ in range(steps):
        dw, db = gradients(x, y, w, b)
        w -= lr * dw  # движение ПРОТИВ градиента
        b -= lr * db
        losses.append(mse(predict(x, w, b), y))
    return w, b, losses


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")  # сохраняем график в файл, окно не требуется
    import matplotlib.pyplot as plt

    print("── 1. Один шаг из текста урока ──")
    w, b, losses = train(np.array([1.0]), np.array([3.0]), lr=0.1, steps=1)
    print(f"  Вес: {w:.1f}, смещение: {b:.1f}, предсказание: {w + b:.1f}")
    print(f"  Loss: {losses[0]:.2f} → {losses[-1]:.2f}")

    # Площади — в сотнях квадратных метров, цены — в условных единицах.
    # Мы как авторы задали правило 2*x + 1. Алгоритму даём только пары x, y.
    x = np.array([0.5, 1.0, 1.5, 2.0])
    y = np.array([2.0, 3.0, 4.0, 5.0])
    print("\n── 2. Обучение на четырёх примерах ──")
    w, b, losses = train(x, y)
    print(f"  Вес: {w:.4f} (скрытое правило: 2), смещение: {b:.4f} (скрытое правило: 1)")
    print(f"  Train loss: {losses[0]:.4f} → {losses[-1]:.8f}")

    unseen = np.array([0.75, 1.75])
    answers = predict(unseen, w, b)  # только прямой проход, веса не меняются
    print("\n  Новые площади, которых не было в train (инференс):")
    for area, answer in zip(unseen, answers, strict=True):
        print(f"  {area * 100:.0f} м² → {answer:.4f}; ожидается по учебному правилу: {2 * area + 1:.2f}")
    print("  Это проверка на паре новых точек, а не доказательство качества реальной модели цен.")

    print("\n── 3. Три скорости обучения: 80 шагов у каждого варианта ──")
    # Расходящийся вариант рисуем отдельно: иначе его огромные числа скроют
    # различие между двумя нормальными кривыми на общей шкале.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for rate in [0.001, 0.1, 1.0]:
        _, _, curve = train(x, y, lr=rate, steps=80)
        print(f"  lr={rate}: loss {curve[0]:.4f} → {curve[-1]:.6g}")
        ax = axes[1] if rate == 1.0 else axes[0]
        ax.semilogy(curve, label=f"learning rate = {rate}")
    axes[0].set_title("Маленький и подходящий шаг")
    axes[1].set_title("Слишком большой шаг — ошибка растёт")
    for ax in axes:
        ax.set_xlabel("Шаг обновления")
        ax.set_ylabel("MSE (логарифмическая шкала)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()

    output = Path(__file__).parent / "out" / "02_learning_rates.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик: {output}")
    print(f"Открыть на Mac: open {output}")
    print("Скорость обучения — не температура генерации: это разные этапы и разные ручки.")


if __name__ == "__main__":
    main()
