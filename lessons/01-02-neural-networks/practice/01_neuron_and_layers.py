"""Урок 1.2 · Устройство нейрона и зачем сети нелинейность.

Запуск из корня проекта:
    uv run python lessons/01-02-neural-networks/practice/01_neuron_and_layers.py

Веса заданы вручную — обучения здесь нет. Его посмотрим в 02_learning.py.
"""

from collections.abc import Sequence

import numpy as np


def neuron(features: Sequence[float], weights: Sequence[float], bias: float) -> float:
    """Взвешенная сумма → смещение → ReLU."""
    score = float(np.dot(features, weights)) + bias
    return max(0.0, score)


def xor_network(inputs: np.ndarray, *, use_relu: bool = True) -> np.ndarray:
    """Два входа → два скрытых нейрона с ReLU → один линейный выход.

    Первый скрытый нейрон: ReLU(a + b).
    Второй:                ReLU(a + b - 1).
    Выход:                 первый - 2 * второй.

    Для бинарных входов 00, 01, 10, 11 получаем 0, 1, 1, 0.
    """
    weights = np.array([[1.0, 1.0], [1.0, 1.0]])  # строки — скрытые нейроны
    bias = np.array([0.0, -1.0])
    hidden = inputs @ weights.T + bias  # строки inputs — разные примеры
    if use_relu:
        hidden = np.maximum(0, hidden)
    return hidden @ np.array([1.0, -2.0])  # на выходе активации нет, смещение равно нулю


def main() -> None:
    print("── 1. Тот же «салатный» нейрон из урока 1.1, но со смещением и ReLU ──")
    weights = [0.1, 0.3, 0.8]
    bias = -5.0
    for name, fruit in {"яблоко": [6, 5, 8], "банан": [8, 1, 1]}.items():
        before_bias = float(np.dot(fruit, weights))
        print(f"  {name}: сумма {before_bias:.1f} → со смещением {before_bias + bias:.1f} → после ReLU {neuron(fruit, weights, bias):.1f}")

    print("\n── 2. Два датчика: сработал ровно один (XOR) ──")
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    expected = [0, 1, 1, 0]
    nonlinear = xor_network(inputs)
    linear = xor_network(inputs, use_relu=False)
    print("  входы | правильный ответ | с ReLU | без ReLU")
    for pair, target, good, bad in zip(inputs, expected, nonlinear, linear, strict=True):
        print(f"   {int(pair[0])}{int(pair[1])}   |        {target}         |   {good:.0f}    |    {bad:.0f}")

    print(
        "\nБез ReLU те же слои дают правило «2 - a - b», а не XOR.\n"
        "Это пример одной настройки. Более общее утверждение из урока:\n"
        "никакая одна прямая не разделит противоположные углы квадрата.\n"
        "\nВажно: веса мы придумали сами. Этот опыт показывает возможности\n"
        "архитектуры, а не то, как обучение находит веса."
    )


if __name__ == "__main__":
    main()
