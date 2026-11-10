"""Урок 0.3 · Эксперимент 1 — первый вызов LLM из Python.

Здесь всё «голыми руками», без вспомогательных модулей, чтобы был виден каждый шаг.

Запуск из корня репозитория:
    uv run python lessons/00-03-first-llm-call/practice/01_first_call.py

Можно запускать по ячейкам (# %%) в Zed REPL или Jupyter.
"""

# %% 1. Подключаемся: три настройки «универсальной розетки»
import os
import time

from dotenv import find_dotenv, load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

load_dotenv(find_dotenv(usecwd=True))  # читаем ключи из .env в корне проекта

# client = OpenAI(
#     base_url="http://localhost:11434/v1",  # 1) адрес: локальный Ollama
#     api_key="ollama",  # 2) ключ: Ollama его не проверяет, но SDK требует строку
# )
# MODEL = os.getenv("MODEL_LOCAL", "qwen3.5:4b")  # 3) модель

# --- Облачная модель Gemini; сам ключ никогда не печатаем. ---
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
    timeout=30.0,  # ожидание ответа, а не стандартные 600 секунд SDK
    max_retries=0,  # в первом эксперименте показываем ошибку сразу, без скрытых повторов
)
MODEL = os.getenv("MODEL_GEMINI", "gemini-3.1-flash-lite")

# %% 2. Запрос: модель + сообщения с ролями + параметры
print(f"⏳ Отправляю запрос к {MODEL}; тайм-аут ожидания 30 секунд…", flush=True)
started = time.perf_counter()
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ты — лаконичный преподаватель. Отвечай по-русски, 2–3 предложения."},
            {"role": "user", "content": "Что такое токен в языковой модели?"},
        ],
        temperature=0.7,
        max_tokens=2000,  # с запасом: reasoning-модели тратят токены ещё и на «размышления»
    )
except APITimeoutError:
    print("❌ Тайм-аут: ответ не пришёл вовремя. Повтори позже или выбери другую MODEL_GEMINI в .env.")
    raise SystemExit(1) from None
except APIConnectionError:
    print("❌ Нет соединения с API. Проверь интернет и настройки сети.")
    raise SystemExit(1) from None
except APIStatusError as exc:
    print(f"❌ Провайдер вернул HTTP {exc.status_code}.")
    if exc.status_code >= 500:
        print("Сервис или модель временно недоступны. Повтори позже или выбери другую MODEL_GEMINI в .env.")
    elif exc.status_code == 429:
        print("Превышена квота или частота запросов. Проверь лимиты Gemini API и повтори позже.")
    elif exc.status_code == 404:
        print("Не найден endpoint или модель. Проверь адрес API и MODEL_GEMINI в .env.")
    elif exc.status_code in (401, 403):
        print("Нет доступа. Проверь ключ и разрешения проекта; сам ключ никому не отправляй.")
    raise SystemExit(1) from None
print(f"✅ Ответ получен за {time.perf_counter() - started:.1f} с", flush=True)

# %% 3. Разбираем ответ
choice = response.choices[0]
usage = response.usage

reasoning = getattr(choice.message, "reasoning", None)  # «размышления», если модель их отдаёт
if reasoning:
    print(f"💭 Размышления ({len(reasoning)} символов):\n{reasoning[:500]}...\n")

print("🤖 Ответ:\n" + (choice.message.content or "<пусто>"))
print("\n🏁 finish_reason:", choice.finish_reason, " ← stop = закончил сам, length = обрезан лимитом")
if usage is not None:
    print(f"🧾 Токены: вход {usage.prompt_tokens} | выход {usage.completion_tokens} | всего {usage.total_tokens}")
else:
    print("🧾 Провайдер не вернул счётчик токенов")

# %% 4. «Сырой» ответ целиком — посмотри на структуру JSON
print(response.model_dump_json(indent=2, exclude_none=True))
