"""Урок 2.2 · Запросы, роли и история простого текстового Chat API.

По умолчанию только синтетический preview, БЕЗ сети и чтения .env:
    uv run python lessons/02-02-chat-api/practice/01_chat_api.py

Настоящие вызовы только явно:
    .../01_chat_api.py --live --provider gemini
    .../01_chat_api.py --live --provider local --model ИМЯ_МОДЕЛИ

Живой режим расходует квоту или деньги выбранного сервиса. OpenRouter не нужен.
"""

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

SYSTEM = "Ты — Знайка, учебный помощник. Отвечай кратко по-русски. Не выдумывай отсутствующие внутренние данные."
History = Sequence[tuple[str, str]]  # только завершённые пары, не журнал сетевых событий


def build_messages(system_instruction: str, history: History, user_text: str) -> list[ChatCompletionMessageParam]:
    """Не изменяем исходную историю; роли назначаются здесь, а не парсятся из content."""
    if not system_instruction.strip() or not user_text.strip():
        raise ValueError("Инструкция и новый вопрос не должны быть пустыми")
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_instruction}]
    for previous_user, previous_assistant in history:
        if not previous_user.strip() or not previous_assistant.strip():
            raise ValueError("История содержит незавершённую текстовую пару")
        messages.append({"role": "user", "content": previous_user})
        messages.append({"role": "assistant", "content": previous_assistant})
    messages.append({"role": "user", "content": user_text})
    return messages


def extract_text(response: ChatCompletion) -> str:
    """Узкий контракт: непустой текст с обычным завершением, без refusal и инструментов."""
    if not response.choices:
        raise ValueError("В ответе нет варианта choices")
    choice = response.choices[0]
    message = choice.message
    if message.refusal:
        raise ValueError("Модель вернула отдельный отказ; это не принятая текстовая пара этой лаборатории")
    if choice.finish_reason == "length":
        raise ValueError("Ответ обрезан лимитом (length); не добавляю его как завершённый")
    if message.tool_calls or choice.finish_reason == "tool_calls":
        raise ValueError("Получен вызов инструмента; этот пример его не исполняет, см. урок 2.5")
    if choice.finish_reason != "stop":
        raise ValueError("Нестандартное завершение; требуется отдельная обработка")
    if not isinstance(message.content, str) or not message.content.strip():
        raise ValueError("Нет непустого текстового ответа; None не является репликой")
    return message.content


def chat_turn(
    client: OpenAI, model: str, system_instruction: str, history: History, user_text: str, *, max_tokens: int = 512
) -> tuple[str, list[tuple[str, str]], ChatCompletion]:
    """Одна попытка; новая история возвращается только после подходящего ответа."""
    if not model.strip() or max_tokens <= 0:
        raise ValueError("Нужны модель и положительный лимит выхода")
    messages = build_messages(system_instruction, history, user_text)
    response = client.chat.completions.create(model=model, messages=messages, stream=False, max_tokens=max_tokens)
    text = extract_text(response)
    updated_history = [*history, (user_text, text)]
    return text, updated_history, response


def resolve_provider(provider: str, model_override: str | None, env: Mapping[str, str]) -> tuple[str, str, str]:
    """Адрес, ключ, модель. Не печатай весь возвращённый кортеж: в нём есть секрет."""
    if provider == "gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = env.get("GEMINI_API_KEY", "")
        model_variable = "MODEL_GEMINI"
        if not api_key.strip():
            raise ValueError("Не задан GEMINI_API_KEY; ключ хранится в .env или окружении")
    elif provider == "local":
        base_url, api_key = "http://localhost:11434/v1", "ollama"
        model_variable = "MODEL_LOCAL"
    else:
        raise ValueError("Неизвестный провайдер: ожидается gemini или local")
    model = model_override if model_override is not None else env.get(model_variable, "")
    if not model.strip():
        raise ValueError(f"Укажи --model или {model_variable}; автоматической модели по умолчанию нет")
    return base_url, api_key, model.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Предпросмотр запроса или явный живой текстовый чат")
    parser.add_argument("--live", action="store_true", help="разрешить реальные API-вызовы")
    parser.add_argument("--provider", choices=["gemini", "local"], default="gemini")
    parser.add_argument("--model", help="явная модель вместо MODEL_GEMINI / MODEL_LOCAL")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens должен быть положительным")

    if not args.live:
        print("Без сети: синтетический preview, не ответ модели. .env не читается.\n")
        examples: list[tuple[str, list[tuple[str, str]]]] = [
            ("Без истории", []),
            ("С вымышленной завершённой парой", [("Меня зовут Алекс", "Понял, Алекс")]),
        ]
        for label, history in examples:
            payload = {
                "model": "preview-only-not-an-api-model",
                "messages": build_messages(SYSTEM, history, "Как меня зовут?"),
                "stream": False,
                "max_tokens": args.max_tokens,
            }
            print(label + ":")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\nДля настоящего чата нужен --live. Облачный режим может расходовать деньги или квоту.")
        return

    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")  # только после явного --live; существующее окружение не переопределяем
    try:
        base_url, api_key, model = resolve_provider(args.provider, args.model, os.environ)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    print(f"Живой режим: {args.provider}, модель {model}. Облачные вызовы расходуют квоту или деньги.")
    print("Команды: /reset — очистить историю процесса, /exit — выйти. Ключи и внутренние документы в чат не вставляй.")
    history: list[tuple[str, str]] = []
    try:
        with OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0) as client:
            while True:
                try:
                    user_text = input("Вы > ")
                except EOFError:
                    break
                command = user_text.strip()
                if command == "/exit":
                    break
                if command == "/reset":
                    history = []
                    print("История процесса очищена. Это не удаляет журналы у провайдера.")
                    continue
                if not command:
                    continue
                print(f"Отправляю запрос к {model}; тайм-аут ожидания 30 с, без автоматических повторов…", flush=True)
                try:
                    text, updated, response = chat_turn(client, model, SYSTEM, history, user_text, max_tokens=args.max_tokens)
                except APITimeoutError:
                    print("Тайм-аут. Завершённая история не изменилась; сервер мог уже выполнять вычисление.")
                    continue
                except APIConnectionError:
                    print("Нет соединения. Проверь сеть или запуск Ollama; история не изменилась.")
                    continue
                except APIStatusError as exc:
                    print(f"API вернул HTTP {exc.status_code}. Проверь доступ, модель, квоту или состояние сервиса.")
                    continue
                except ValueError as exc:
                    print(f"Результат не принят: {exc}")
                    continue
                history = updated
                print(f"Знайка > {text}")
                if response.usage is None:
                    print("Usage не получен — это не доказательство нулевого расхода.")
                else:
                    print(f"Токены: вход {response.usage.prompt_tokens}, выход {response.usage.completion_tokens}")
                print(f"Принятых пар в истории: {len(history)}")
    except KeyboardInterrupt:
        print("\nЧат остановлен.")


if __name__ == "__main__":
    main()
