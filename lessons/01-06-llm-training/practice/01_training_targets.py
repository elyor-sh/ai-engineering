"""Урок 1.6 · Целевые токены и маска потери: что именно мы оцениваем.

Запуск:
    uv run python lessons/01-06-llm-training/practice/01_training_targets.py

Условный словарь и фиксированные logits. Никакая модель здесь не обучается.
"""

import numpy as np


def shift_targets(token_ids: list[int], assistant_tokens: list[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вход t предсказывает токен t+1. Маску сдвигаем вместе с ЦЕЛЕВЫМИ токенами."""
    if len(token_ids) < 2 or len(token_ids) != len(assistant_tokens):
        raise ValueError("Нужны минимум два токена и одинаковая длина разметки")
    ids = np.asarray(token_ids, dtype=int)
    mask = np.asarray(assistant_tokens, dtype=bool)
    return ids[:-1], ids[1:], mask[1:]


def masked_cross_entropy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Средний -log p(правильного токена) по выбранным позициям, устойчиво к большим logits."""
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    if logits.ndim != 2 or 0 in logits.shape or not np.isfinite(logits).all():
        raise ValueError("Нужна непустая матрица конечных logits")
    if targets.shape != (logits.shape[0],) or mask.shape != targets.shape:
        raise ValueError("По одному target и значению маски на каждую строку logits")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]) or not mask.any():
        raise ValueError("Некорректный ID цели или нет ни одной оцениваемой позиции")
    selected = logits[mask]
    shifted = selected - selected.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    losses = log_normalizer - shifted[np.arange(selected.shape[0]), targets[mask]]
    return float(losses.mean())


def main() -> None:
    vocabulary = ["<user>", "2+2?", "<assistant>", "4", "<end>", "5"]
    sequence = [0, 1, 2, 3, 4]
    assistant_tokens = [False, False, False, True, True]  # считаем ответ и завершение
    inputs, targets, sft_mask = shift_targets(sequence, assistant_tokens)

    # Задаём p правильной цели в каждой позиции, остаток распределяем поровну.
    # Матрица моделирует готовые предсказания; forward pass и градиентов здесь нет.
    target_probabilities = np.array([0.05, 0.1, 0.8, 0.9])
    probabilities = np.repeat(((1 - target_probabilities) / (len(vocabulary) - 1))[:, None], len(vocabulary), axis=1)
    probabilities[np.arange(len(targets)), targets] = target_probabilities
    logits = np.log(probabilities)

    print("Это учебные ID и маркеры, не настоящий chat template Gemini или OpenAI.\n")
    print("Входная позиция → следующая цель | p цели | loss | включена в SFT-loss")
    for source, target, probability, selected in zip(inputs, targets, target_probabilities, sft_mask, strict=True):
        flag = "да" if selected else "нет"
        print(f"  {vocabulary[source]:12} → {vocabulary[target]:12} | {probability:.2f} | {-np.log(probability):.3f} | {flag}")

    full_loss = masked_cross_entropy(logits, targets, np.ones_like(sft_mask))
    sft_loss = masked_cross_entropy(logits, targets, sft_mask)
    print(f"\nПотеря по всем следующим токенам: {full_loss:.4f} (4 позиции)")
    print(f"Потеря по ответу ассистента:     {sft_loss:.4f} (2 позиции)")
    print("Число изменилось, хотя предсказания и параметры не улучшались: поменяли оцениваемые позиции.")

    changed = logits.copy()
    changed[:2] = -100.0
    changed[0, targets[0]] = 100.0
    changed[1, targets[1]] = 100.0
    changed_sft_loss = masked_cross_entropy(changed, targets, sft_mask)
    print(f"\nИсправили только предсказания до ответа: SFT-loss {sft_loss:.4f} → {changed_sft_loss:.4f}")
    print("Вопрос остаётся контекстом. Маска потери не запрещает attention видеть его.")
    print("Это не сравнение качества двух обученных моделей и не реализация полноценного SFT.")


if __name__ == "__main__":
    main()
