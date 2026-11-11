"""Урок 1.3 · Бюджет контекста и стоимость по условному тарифу.

Запуск:
    uv run python lessons/01-03-tokenization/practice/03_context_budget.py

Только арифметика. Никаких API-запросов и чтения ключей из .env.
Модель условная: здесь предполагается ОБЩИЙ лимит на вход + выход.
"""


def remaining_context(context_limit: int, input_tokens: int, output_reserved: int, *, margin: int = 0) -> int:
    """Остаток для дополнительных данных; отрицательное число означает переполнение."""
    if min(context_limit, input_tokens, output_reserved, margin) < 0:
        raise ValueError("Размеры и запас должны быть неотрицательными")
    return context_limit - input_tokens - output_reserved - margin


def cost_usd(input_tokens: int, output_tokens: int, input_per_million: float, output_per_million: float) -> float:
    """Цена для простого тарифа без кэширования, отдельных reasoning-категорий и сборов."""
    if min(input_tokens, output_tokens, input_per_million, output_per_million) < 0:
        raise ValueError("Числа токенов и тарифы должны быть неотрицательными")
    return (input_tokens * input_per_million + output_tokens * output_per_million) / 1_000_000


def main() -> None:
    window = 8192
    instructions_and_format = 500  # служебный формат в этом примере УЖЕ включён
    history = 1500
    question = 200
    output_reserved = 1000
    margin = 300
    base_input = instructions_and_format + history + question
    documents = [("Документ A", 1800), ("Документ B", 1600), ("Документ C", 1500)]

    available = remaining_context(window, base_input, output_reserved, margin=margin)
    print(f"Общее окно: {window}; базовый вход: {base_input}; резерв ответа: {output_reserved}; запас: {margin}")
    print(f"Документам доступно: {available} токенов")
    document_tokens = sum(size for _, size in documents)
    print(f"Три найденных фрагмента: {documents}; всего {document_tokens} токенов")
    remaining = remaining_context(window, base_input + document_tokens, output_reserved, margin=margin)
    print(f"Остаток: {remaining} → переполнение на {-remaining} токенов")

    # Учебное решение: убираем последний фрагмент. В RAG нужно выбирать по полезности,
    # учитывать источники и проверять качество, а не всегда отрезать последний документ.
    selected = documents[:-1]
    final_input = base_input + sum(size for _, size in selected)
    print(f"\nОставили два фрагмента. Вход теперь: {final_input} токенов")
    print("Остаток с учётом всех резервов:", remaining_context(window, final_input, output_reserved, margin=margin))

    # Выдуманные цены для арифметики, не прайс Gemini и не расчёт реального баланса.
    actual_output = 1000
    cost = cost_usd(final_input, actual_output, input_per_million=1.0, output_per_million=5.0)
    print("\nУсловный тариф: $1 / 1М входных, $5 / 1М выходных")
    print(f"При фактическом выходе {actual_output}: ${cost:.4f} за запрос, ${cost * 10_000:.2f} за 10 000 запросов")
    print("Резерв ответа — место, а не гарантированный расход. Если ответ короче, выходных токенов меньше.")


if __name__ == "__main__":
    main()
