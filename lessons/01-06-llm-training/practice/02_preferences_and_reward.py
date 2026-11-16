"""Урок 1.6 · Предпочтение не равно истине: специальный двухответный случай DPO.

Запуск:
    uv run python lessons/01-06-llm-training/practice/02_preferences_and_reward.py

У политики два фиксированных ответа и один параметр theta: это НЕ LLM.
Обновляем theta аналитическим градиентом парной цели с фиксированной опорой.
Нет reward model, PPO, генерации текста, API-запросов и обучения нейросети.
"""

from pathlib import Path

import numpy as np


def answer_logprobs(theta: float) -> np.ndarray:
    """Логарифмы вероятностей двух ответов: log sigmoid(theta), log sigmoid(-theta)."""
    if not np.isfinite(theta):
        raise ValueError("theta должен быть конечным числом")
    return np.array([-np.logaddexp(0.0, -theta), -np.logaddexp(0.0, theta)])


def dpo_pair_loss(policy: np.ndarray, reference: np.ndarray, *, chosen: int, beta: float = 0.5) -> float:
    """Парная цель через изменение log-отношения chosen/rejected относительно опоры.

    policy и reference содержат log-вероятности двух целых фиксированных ответов.
    В LLM вероятности последовательностей связаны с вероятностями их токенов.
    """
    policy, reference = np.asarray(policy, dtype=float), np.asarray(reference, dtype=float)
    if policy.shape != (2,) or reference.shape != (2,) or chosen not in (0, 1):
        raise ValueError("Нужны две log-вероятности в каждой модели и chosen 0 или 1")
    if not np.isfinite(policy).all() or not np.isfinite(reference).all() or not np.isfinite(beta) or beta <= 0:
        raise ValueError("Значения должны быть конечными, beta — положительным")
    rejected = 1 - chosen
    margin = (policy[chosen] - policy[rejected]) - (reference[chosen] - reference[rejected])
    return float(np.logaddexp(0.0, -beta * margin))


def train_toy_policy(
    *, chosen: int, steps: int = 80, lr: float = 0.4, beta: float = 0.5, reference_theta: float = 0.0
) -> tuple[float, np.ndarray, np.ndarray]:
    """Возвращает theta, историю p(ответа 0) и историю loss.

    В этой бинарной политике log(p0/p1) = theta, поэтому градиент можно записать явно.
    Reference фиксируется один раз. Никаких сведений об истинности train не получает.
    """
    if steps < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Число шагов неотрицательное, learning rate положительный и конечный")
    theta = reference_theta
    reference = answer_logprobs(reference_theta)
    policy = answer_logprobs(theta)
    losses = [dpo_pair_loss(policy, reference, chosen=chosen, beta=beta)]
    probabilities = [np.exp(policy[0])]
    direction = 1 if chosen == 0 else -1
    for _ in range(steps):
        scaled_margin = beta * direction * (theta - reference_theta)
        gradient = -beta * direction * np.exp(-np.logaddexp(0.0, scaled_margin))
        theta -= lr * gradient
        policy = answer_logprobs(theta)
        losses.append(dpo_pair_loss(policy, reference, chosen=chosen, beta=beta))
        probabilities.append(np.exp(policy[0]))
    return theta, np.asarray(probabilities), np.asarray(losses)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    answers = [
        "По учебному регламенту: 14 дней.",
        "Я абсолютно уверен: по учебному регламенту разрешено вернуть товар в течение 140 дней, можете смело на это рассчитывать.",
    ]
    print("В искусственном регламенте этого опыта указан срок 14 дней. Это не правовая консультация.")
    print("Два готовых ответа; текст не генерируется:")
    for i, text in enumerate(answers):
        print(f"  {i}: {text}")
    word_counts = [len(text.split()) for text in answers]
    preferred_by_length = max(range(len(answers)), key=lambda i: word_counts[i])
    print(f"\nПлохой критерий «больше слов»: баллы {word_counts} → chosen={preferred_by_length}")
    print("Это нарочно плохая эвристика оценщика, а не обученная reward model.\n")

    scenarios = [("Предпочтён верный ответ", 0, "-"), ("Предпочтён неверный ответ", preferred_by_length, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, chosen, style in scenarios:
        theta, probabilities, losses = train_toy_policy(chosen=chosen)
        print(f"{label}: theta={theta:.3f}")
        print(f"  p(верного ответа): {probabilities[0]:.3f} → {probabilities[-1]:.3f}")
        print(f"  Парная потеря:    {losses[0]:.3f} → {losses[-1]:.3f}")
        axes[0].plot(probabilities, style, label=label)
        axes[1].plot(losses, style, label=label)

    axes[0].set_title("Какой ответ становится вероятнее?")
    axes[0].set_ylabel("Вероятность выбора ответа «14 дней»")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Loss уменьшается в обоих случаях")
    axes[1].set_ylabel("Парная потеря (две совпадающие кривые)")
    for ax in axes:
        ax.set_xlabel("Шаг обновления одного параметра")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(__file__).parent / "out" / "02_preferences_and_reward.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик: {path}")
    print("Оптимизация следует предпочтениям, а не самостоятельно проверяет регламент.")
    print("Это специальный числовой случай DPO-цели, не тренировка настоящей языковой модели.")


if __name__ == "__main__":
    main()
