"""Урок 0.3 · Эксперимент 2 — модель ничего не помнит.

Запуск:
    uv run python lessons/00-03-first-llm-call/practice/02_memory.py
    uv run python lessons/00-03-first-llm-call/practice/02_memory.py --provider fast
"""

import argparse

from openai.types.chat import ChatCompletionMessageParam

from providers import PROVIDERS, ask, explain_error

parser = argparse.ArgumentParser()
parser.add_argument("--provider", default="local", choices=list(PROVIDERS))
provider = PROVIDERS[parser.parse_args().provider]
print(f"Модель: {provider.label} · {provider.model}\n")

SYSTEM: ChatCompletionMessageParam = {"role": "system", "content": "Отвечай кратко, одним-двумя предложениями, по-русски."}
INTRO: ChatCompletionMessageParam = {"role": "user", "content": "Привет! Меня зовут Алекс, я учу AI-инженерию."}
QUESTION: ChatCompletionMessageParam = {"role": "user", "content": "Как меня зовут?"}

try:
    # ── Вызов 1: представляемся ──
    a1 = ask(provider, [SYSTEM, INTRO])
    print("① Представились →", a1.text, "\n")

    # ── Вызов 2: новый запрос БЕЗ истории ──
    a2 = ask(provider, [SYSTEM, QUESTION])
    print("② Спросили имя БЕЗ истории →", a2.text)
    print(f"   входных токенов: {a2.prompt_tokens}\n")

    # ── Вызов 3: тот же вопрос, но С историей ──
    with_history: list[ChatCompletionMessageParam] = [SYSTEM, INTRO, {"role": "assistant", "content": a1.text}, QUESTION]
    a3 = ask(provider, with_history)
    print("③ Спросили имя С историей →", a3.text)
    print(f"   входных токенов: {a3.prompt_tokens}  (память стоила +{a3.prompt_tokens - a2.prompt_tokens})\n")

    # ── Бонус: подделываем историю ──
    # Модель видит только то, что мы ей прислали. Что, если «её» прошлый ответ был другим?
    forged: list[ChatCompletionMessageParam] = [
        SYSTEM, INTRO, {"role": "assistant", "content": "Приятно познакомиться, Борис!"}, QUESTION
    ]
    a4 = ask(provider, forged)
    print("④ Подделанная история (модель «назвала» нас Борисом) →", a4.text)
    print("\n🤔 Поверила ли модель подделке? Почему это может быть и полезно, и опасно?")
except Exception as e:  # noqa: BLE001 — учебный скрипт: показываем любую ошибку понятным текстом
    print("❌", explain_error(provider, e))
