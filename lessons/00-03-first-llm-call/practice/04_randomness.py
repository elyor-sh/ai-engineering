"""Урок 0.3 · Эксперимент 4 — один вопрос, разные ответы (недетерминированность).

Запуск:
    uv run python lessons/00-03-first-llm-call/practice/04_randomness.py
    uv run python lessons/00-03-first-llm-call/practice/04_randomness.py --provider fast --runs 5

⚠️ Многие современные модели (например, Claude и GPT через OpenRouter) игнорируют temperature.
   Для этого эксперимента выбирай local, fast или gemini.
"""

import argparse

from providers import PROVIDERS, ask, explain_error

PROMPT = "Придумай одно оригинальное название для кофейни. Ответь только названием, без пояснений."

parser = argparse.ArgumentParser()
parser.add_argument("--provider", default="local", choices=list(PROVIDERS))
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()
provider = PROVIDERS[args.provider]

print(f"Модель: {provider.label} · {provider.model}")
print(f"Вопрос: {PROMPT}\n")

for temperature in (1.0, 0.0):
    print(f"── temperature = {temperature} ──")
    results = []
    for i in range(1, args.runs + 1):
        try:
            a = ask(provider, [{"role": "user", "content": PROMPT}], temperature=temperature, max_tokens=2000)
        except Exception as e:  # noqa: BLE001 — учебный скрипт
            print("❌", explain_error(provider, e))
            raise SystemExit(1) from e
        results.append(a.text.strip())
        print(f"  {i}. {a.text.strip()}")
    unique = len(set(results))
    print(f"  → уникальных ответов: {unique} из {args.runs}\n")

print("🤔 При какой температуре ответы стабильнее? Какую выбрал бы для извлечения данных, а какую — для генерации идей?")
