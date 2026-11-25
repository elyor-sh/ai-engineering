"""Урок 2.1 · Сначала допустимость, затем компромисс качества, цены и задержки.

Запуск:
    uv run python lessons/02-01-model-landscape/practice/01_model_selection.py

ВСЕ профили и метрики вымышлены. Это не рейтинг рынка и не выбор реальной API-модели.
Нет запросов, чтения .env или проверки действительных лицензий и маршрутов данных.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class Candidate:
    name: str
    hosting: Literal["local", "cloud"]
    quality: float  # условная доля успеха на одном и том же учебном наборе
    p95_seconds: float
    monthly_cost: float  # условные полные затраты на одну и ту же нагрузку
    input_limit: int
    output_limit: int
    context_limit: int  # в этой вымышленной конфигурации общий предел входа и выхода
    supports_structured_output: bool

    def __post_init__(self) -> None:
        values = (self.quality, self.p95_seconds, self.monthly_cost, self.input_limit, self.output_limit, self.context_limit)
        if any(not math.isfinite(value) or value < 0 for value in values) or not 0 <= self.quality <= 1:
            raise ValueError("Метрики должны быть конечными, неотрицательными; качество — в [0, 1]")
        if min(self.input_limit, self.output_limit, self.context_limit) <= 0 or self.hosting not in ("local", "cloud"):
            raise ValueError("Нужны положительные лимиты и известный тип размещения")


@dataclass(frozen=True)
class Requirements:
    min_quality: float
    max_p95_seconds: float
    max_monthly_cost: float
    input_tokens: int  # полный вход с оговорённым учётом служебных данных
    output_tokens: int  # резерв выхода, а не обещание фактической длины
    allow_cloud: bool
    require_structured_output: bool

    def __post_init__(self) -> None:
        values = (self.min_quality, self.max_p95_seconds, self.max_monthly_cost, self.input_tokens, self.output_tokens)
        if any(not math.isfinite(value) or value < 0 for value in values) or not 0 <= self.min_quality <= 1:
            raise ValueError("Требования должны быть конечными и неотрицательными; доля качества — в [0, 1]")


def demo_candidates() -> list[Candidate]:
    """Одна вымышленная задача и нагрузка, не характеристики Gemini, Claude или локальной LLM."""
    return [
        Candidate("demo-local", "local", 0.86, 6.0, 50.0, 8192, 2048, 8192, False),
        Candidate("demo-cloud-fast", "cloud", 0.91, 2.0, 540.0, 32768, 4096, 32768, True),
        Candidate("demo-cloud-large", "cloud", 0.96, 8.0, 2500.0, 131072, 8192, 131072, True),
        Candidate("demo-cloud-alternate", "cloud", 0.89, 3.0, 700.0, 32768, 2048, 32768, True),
    ]


def rejection_reasons(candidate: Candidate, requirements: Requirements) -> list[str]:
    """Возвращаем объяснения, а не компенсируем обязательные условия общим баллом."""
    reasons = []
    if candidate.hosting == "cloud" and not requirements.allow_cloud:
        reasons.append("облачная передача запрещена")
    if candidate.quality < requirements.min_quality:
        reasons.append("качество ниже порога")
    if candidate.p95_seconds > requirements.max_p95_seconds:
        reasons.append("p95 выше лимита")
    if candidate.monthly_cost > requirements.max_monthly_cost:
        reasons.append("месячная стоимость выше бюджета")
    if requirements.input_tokens > candidate.input_limit:
        reasons.append("вход не помещается")
    if requirements.output_tokens > candidate.output_limit:
        reasons.append("резерв выхода превышает лимит")
    if requirements.input_tokens + requirements.output_tokens > candidate.context_limit:
        reasons.append("общий бюджет контекста превышен")
    if requirements.require_structured_output and not candidate.supports_structured_output:
        reasons.append("нет обязательного структурированного вывода")
    return reasons


def shortlist(candidates: Sequence[Candidate], requirements: Requirements) -> list[Candidate]:
    """Пустой результат — нормальный исход. Никакого автоматического обхода ограничений."""
    return [candidate for candidate in candidates if not rejection_reasons(candidate, requirements)]


def dominates(first: Candidate, second: Candidate) -> bool:
    """Только качество ↑, p95 ↓ и полная месячная цена ↓. Другие свойства здесь не ранжируем."""
    no_worse = first.quality >= second.quality and first.p95_seconds <= second.p95_seconds and first.monthly_cost <= second.monthly_cost
    strictly_better = first.quality > second.quality or first.p95_seconds < second.p95_seconds or first.monthly_cost < second.monthly_cost
    return no_worse and strictly_better


def pareto_front(feasible_candidates: Sequence[Candidate]) -> list[Candidate]:
    """Ожидает уже допустимых кандидатов: фильтры и граница Парето — разные шаги."""
    return [candidate for candidate in feasible_candidates if not any(dominates(other, candidate) for other in feasible_candidates)]


def main() -> None:
    candidates = demo_candidates()
    strict = Requirements(0.90, 5.0, 1000.0, 10_000, 1000, True, True)
    print("ВСЕ числа и имена вымышлены. Не используй demo-* как идентификаторы настоящего API.\n")
    print("Профиль | качество | p95, с | полная месячная цена")
    for candidate in candidates:
        print(f"{candidate.name:22} | {candidate.quality:.2f} | {candidate.p95_seconds:6.1f} | ${candidate.monthly_cost:.0f}")

    print("\nСценарий 1: качество ≥0.90, p95 ≤5 с, бюджет ≤$1000, вход 10000, выход 1000, нужен структурированный вывод.")
    for candidate in candidates:
        reasons = rejection_reasons(candidate, strict)
        print(f"  {candidate.name}: " + ("; ".join(reasons) if reasons else "допустим"))
    print("Короткий список:", [candidate.name for candidate in shortlist(candidates, strict)])

    print("\nСценарий 2: те же требования, но передача в облако запрещена.")
    private = replace(strict, allow_cloud=False)
    print("Короткий список:", [candidate.name for candidate in shortlist(candidates, private)])
    print("Если список пуст: уточнить решение и собрать данные. Не включать облачный fallback молча.")

    print("\nСценарий 3: отдельная учебная задача с более мягкими требованиями.")
    relaxed = replace(
        strict,
        min_quality=0.85,
        max_p95_seconds=10,
        max_monthly_cost=3000,
        input_tokens=6000,
        require_structured_output=False,
    )
    feasible = shortlist(candidates, relaxed)
    front = pareto_front(feasible)
    print("Допустимые:", [candidate.name for candidate in feasible])
    print("Граница Парето:", [candidate.name for candidate in front])
    print("demo-cloud-fast доминирует demo-cloud-alternate по этим трём метрикам.")
    print("Это не автоматический выбор победителя и не оценка реальных лицензий, данных или качества.")


if __name__ == "__main__":
    main()
