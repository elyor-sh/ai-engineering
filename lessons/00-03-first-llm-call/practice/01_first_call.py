"""Урок 0.3 · Эксперимент 1 — первый вызов LLM из Python.

Здесь всё «голыми руками», без вспомогательных модулей, чтобы был виден каждый шаг.

Запуск из корня репозитория:
    uv run python lessons/00-03-first-llm-call/practice/01_first_call.py

Можно запускать по ячейкам (# %%) в Zed REPL или Jupyter.
"""

# %% 1. Подключаемся: три настройки «универсальной розетки»
import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))  # читаем ключи из .env в корне проекта

client = OpenAI(
    base_url="http://localhost:11434/v1",  # 1) адрес: локальный Ollama
    api_key="ollama",  # 2) ключ: Ollama его не проверяет, но SDK требует строку
)
MODEL = os.getenv("MODEL_LOCAL", "qwen3.5:4b")  # 3) модель

# --- Чтобы спросить облачную модель, раскомментируй (и закомментируй блок выше): ---
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ["OPENROUTER_API_KEY"],
# )
# MODEL = "qwen/qwen3.7-flash"

# %% 2. Запрос: модель + сообщения с ролями + параметры
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Ты — лаконичный преподаватель. Отвечай по-русски, 2–3 предложения."},
        {"role": "user", "content": "Что такое токен в языковой модели?"},
    ],
    temperature=0.7,
    max_tokens=2000,  # с запасом: reasoning-модели тратят токены ещё и на «размышления»
)

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
