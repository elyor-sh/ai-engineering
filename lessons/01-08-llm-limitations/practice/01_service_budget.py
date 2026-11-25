"""Урок 1.8 · Цена, дополнительные попытки, хвост задержек и условное качество.

Запуск:
    uv run python lessons/01-08-llm-limitations/practice/01_service_budget.py

Только стандартная библиотека, выдуманные тарифы и синтетические наблюдения.
Не вызывает API, не читает .env и не измеряет производительность настоящей модели.
"""

import math
from collections.abc import Sequence
from statistics import mean, median


def price_per_call(input_tokens: int, output_tokens: int, input_rate: float, output_rate: float) -> float:
    """Тарифы в долларах за миллион токенов; без кэширования и других категорий оплаты."""
    if any(not math.isfinite(value) or value < 0 for value in (input_tokens, output_tokens, input_rate, output_rate)):
        raise ValueError("Числа токенов и тарифы должны быть неотрицательными и конечными")
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def monthly_budget(daily_requests: int, days: int, call_price: float, *, extra_attempts_per_request: float = 0.0) -> float:
    """Среднее число ДОПОЛНИТЕЛЬНЫХ оплачиваемых вызовов на один исходный запрос.

    В этой оценке у дополнительной попытки та же средняя цена, что у основной.
    Это допущение задачи, а не утверждение о биллинге ошибок любого провайдера.
    """
    values = (daily_requests, days, call_price, extra_attempts_per_request)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Трафик, дни, цена и число дополнительных попыток неотрицательные и конечные")
    return daily_requests * days * call_price * (1 + extra_attempts_per_request)


def nearest_rank_percentile(latencies: Sequence[float], fraction: float) -> float:
    """Процентиль методом ближайшего ранга: ceil(N * fraction), ранги начинаются с 1.

    Это один определённый метод. Другие инструменты могут интерполировать значения.
    """
    if not latencies or any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("Нужна непустая последовательность конечных неотрицательных задержек")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Доля процентиля должна быть в (0, 1]")
    return sorted(latencies)[math.ceil(len(latencies) * fraction) - 1]


def latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    """Среднее, обычная медиана и p95 ближайшего ранга; валидация через percentile."""
    p95 = nearest_rank_percentile(latencies, 0.95)
    return {"mean": mean(latencies), "median": median(latencies), "p95": p95}


def conditional_chain_success(conditional_rates: Sequence[float]) -> float:
    """Перемножаем УСЛОВНЫЕ доли успеха каждого этапа при успехе предыдущих.

    Функция не может проверить, что предоставленные доли действительно условные.
    Нельзя подставлять произвольные несогласованные метрики из разных отчётов.
    """
    if not conditional_rates or any(not math.isfinite(rate) or not 0 <= rate <= 1 for rate in conditional_rates):
        raise ValueError("Нужны условные доли в [0, 1]")
    return math.prod(conditional_rates)


def main() -> None:
    print("Все тарифы и наблюдения искусственные. Это не счёт Gemini и не замер сервера.\n")
    input_tokens, output_tokens = 2000, 400
    input_rate, output_rate = 0.5, 2.0
    daily_requests, days = 10_000, 30
    call_price = price_per_call(input_tokens, output_tokens, input_rate, output_rate)
    baseline = monthly_budget(daily_requests, days, call_price)
    with_extras = monthly_budget(daily_requests, days, call_price, extra_attempts_per_request=0.1)
    print("── Цена всей нагрузки ──")
    print(f"  Вызов: {input_tokens} входных, {output_tokens} выходных; тарифы ${input_rate} / ${output_rate} за 1М")
    print(f"  Цена одного вызова: ${call_price:.4f}")
    print(f"  {daily_requests} исходных запросов в день × {days} дней: ${baseline:.2f}")
    print(f"  + одна дополнительная одинаково оплачиваемая попытка на 10 исходных: ${with_extras:.2f}")
    print("  Кэш, другие API, налоги и работа человека здесь не учитываются.")

    latencies = [1.0] * 18 + [15.0] * 2
    stats = latency_summary(latencies)
    print("\n── Среднее и хвост ──")
    print("  18 наблюдений по 1 с, 2 по 15 с")
    print(f"  Среднее: {stats['mean']:.1f} с; медиана: {stats['median']:.1f} с; p95: {stats['p95']:.1f} с")
    print("  p95 — метод ближайшего ранга. Все эти наблюдения синтетические.")
    print("  В настоящей системе нельзя молча исключать тайм-ауты и показывать только успешные ответы.")

    rates = [0.95, 0.95, 0.95]
    end_to_end = conditional_chain_success(rates)
    print("\n── Качество цепочки ──")
    print("  Подходящий источник: 95%.")
    print("  Верный ответ ПРИ подходящем источнике: 95%.")
    print("  Подходящий формат ПРИ выполнении первых двух условий: 95%.")
    print(f"  Общий успех по этим допущениям: {end_to_end:.4%}")
    print("  Это условные доли, не три независимые метрики, которые можно произвольно перемножить.")
    print("\nПравильная арифметика не подтверждает правильность исходных данных и допущений.")


if __name__ == "__main__":
    main()
