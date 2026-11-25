"""Урок 2.4 · JSON, схема Pydantic и отдельно известные факты.

Без сети и чтения .env:
    uv run python lessons/02-04-structured-outputs/practice/01_structured_outputs.py
    .../01_structured_outputs.py --show-schema

Один реальный запрос только явно, если режим поддерживается моделью и endpoint:
    .../01_structured_outputs.py --live --provider gemini --mode schema

Все исходные заявки и ответы синтетические. Ничего во внешних системах не создаётся.
"""

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NoReturn

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject, ResponseFormatJSONSchema
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SOURCE_TEXT = "Учебная заявка №28: не могу войти в портал. Контактный email не указан."
EXPECTED_REQUEST_ID = 28  # независимая учебная разметка, не ответ другой LLM
EXPECTED_EMAIL = None
Category = Literal["access", "billing", "other", "unknown"]


class TicketDraft(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    request_id: int = Field(description="Идентификатор заявки из источника, не придумывать")
    category: Category = Field(description="access — доступ, billing — оплата, other — другое, unknown — недостаточно данных")
    contact_email: str | None = Field(description="Контакт из источника; null, если он не указан. Ключ обязателен.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повторяющийся ключ JSON запрещён учебным контрактом")
        result[key] = value
    return result


def _reject_constant(_value: str) -> NoReturn:
    raise ValueError("NaN и Infinity не разрешены в JSON этого контракта")


def parse_draft(raw: str) -> TicketDraft:
    """Сначала JSON с уникальными ключами, затем строгая структура. Никакого eval или ремонта."""
    data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    return TicketDraft.model_validate(data)


def verify_known_facts(draft: TicketDraft, *, expected_request_id: int, expected_email: str | None) -> None:
    """Проверяет только два независимо заданных факта. НЕ проверяет семантику category или права."""
    if expected_request_id <= 0 or draft.request_id <= 0:
        raise ValueError("В этой учебной задаче ID должен быть положительным")
    if draft.request_id != expected_request_id:
        raise ValueError("ID не совпадает с независимой разметкой источника")
    if draft.contact_email != expected_email:
        raise ValueError("Контакт не совпадает с независимо известными данными; не принимаем выдумку")


def safe_validation_errors(error: ValidationError) -> list[str]:
    """Только известное имя поля и тип ошибки. Не выводим input, msg, ctx или неизвестные ключи."""
    reports = []
    for item in error.errors(include_input=False, include_url=False):
        location = item["loc"]
        if not location:
            field = "$"
        elif location[0] in TicketDraft.model_fields:
            field = str(location[0])
        else:
            field = "<неизвестное поле>"
        reports.append(f"{field}: {item['type']}")
    return reports


def make_response_format(mode: str) -> ResponseFormatJSONObject | ResponseFormatJSONSchema | None:
    """Контракт совместимого Chat Completions; его поддержка НЕ предполагается у всех серверов."""
    if mode == "prompt":
        return None
    if mode == "json":
        return {"type": "json_object"}
    if mode == "schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "ticket_draft_v1", "strict": True, "schema": TicketDraft.model_json_schema()},
        }
    raise ValueError("Режим должен быть prompt, json или schema")


def request_completion(client: OpenAI, model: str, mode: str, *, source_text: str = SOURCE_TEXT, max_tokens: int = 512) -> ChatCompletion:
    if not model.strip() or not source_text.strip() or max_tokens <= 0:
        raise ValueError("Нужны модель, источник и положительный лимит")
    response_format = make_response_format(mode)
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "Извлеки черновик из учебного источника. Верни только JSON с request_id (целое число), "
                "category (access, billing, other или unknown), contact_email (строка или null). "
                "access означает доступ/вход, billing — оплату. Если категория неясна, unknown. "
                "Если контакт не указан, верни null. Не выдумывай данные и не добавляй других ключей."
            ),
        },
        {"role": "user", "content": source_text},
    ]
    if response_format is None:
        return client.chat.completions.create(model=model, messages=messages, stream=False, max_tokens=max_tokens)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        response_format=response_format,
    )


def extract_json_text(response: ChatCompletion) -> str:
    """Метаданные проверяются ДО разбора JSON: schema не отменяет отказ или обрезку."""
    if not response.choices:
        raise ValueError("Нет варианта ответа")
    choice = response.choices[0]
    message = choice.message
    if message.refusal:
        raise ValueError("Получен отдельный отказ; это не объект нашей схемы")
    if message.tool_calls or choice.finish_reason == "tool_calls":
        raise ValueError("Получен запрос инструмента, который эта лаборатория не исполняет")
    if choice.finish_reason != "stop":
        raise ValueError("Не подтверждено обычное завершение; length и другие исходы не исправляем вслепую")
    if not isinstance(message.content, str) or not message.content.strip():
        raise ValueError("Нет непустого JSON-текста")
    return message.content


def resolve_provider(provider: str, model_override: str | None, env: Mapping[str, str]) -> tuple[str, str, str]:
    """Адрес, секрет, модель. Возвращённый кортеж не печатаем."""
    if provider == "gemini":
        url, key, variable = "https://generativelanguage.googleapis.com/v1beta/openai/", env.get("GEMINI_API_KEY", ""), "MODEL_GEMINI"
        if not key.strip():
            raise ValueError("Не задан GEMINI_API_KEY")
    elif provider == "local":
        url, key, variable = "http://localhost:11434/v1", "ollama", "MODEL_LOCAL"
    else:
        raise ValueError("Ожидается gemini или local")
    model = model_override if model_override is not None else env.get(variable, "")
    if not model.strip():
        raise ValueError(f"Укажи --model или {variable}; автоматической модели по умолчанию нет")
    return url, key, model.strip()


def review_candidate(raw: str) -> bool:
    """Сообщаем результат локальных уровней, не разрешаем никаких бизнес-действий."""
    try:
        draft = parse_draft(raw)
    except json.JSONDecodeError:
        print("JSON не разобран: синтаксис или постороннее оформление. Ремонт не выполнялся.")
        return False
    except ValidationError as error:
        print("Структура отклонена:", "; ".join(safe_validation_errors(error)))
        return False
    except ValueError:
        print("JSON отклонён дополнительными правилами: неоднозначные ключи или нестандартные константы.")
        return False
    print("Структура: OK")
    try:
        verify_known_facts(draft, expected_request_id=EXPECTED_REQUEST_ID, expected_email=EXPECTED_EMAIL)
    except ValueError as error:
        print("Два известных факта: НЕ пройдены.", error)
        return False
    print("Два известных факта: совпали.")
    print("Черновик:", draft.model_dump_json())
    print("Смысл category отдельно НЕ подтверждён. Права не проверялись. Никакое действие не выполнялось.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальная валидация или один явный запрос структурированного ответа")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", help="разрешить один настоящий API-вызов")
    mode_group.add_argument("--show-schema", action="store_true", help="показать локальную JSON Schema без сети")
    parser.add_argument("--provider", choices=["gemini", "local"], default="gemini")
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=["prompt", "json", "schema"], default="schema")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("Лимит должен быть положительным")
    if args.show_schema:
        print("Без сети: схема Pydantic, не результат генерации.")
        print(json.dumps(TicketDraft.model_json_schema(), ensure_ascii=False, indent=2))
        return
    if not args.live:
        print("Без сети: синтетические ответы; .env не читается. Это не результаты настоящей LLM.\n")
        print("Источник:", SOURCE_TEXT)
        valid = {"request_id": 28, "category": "access", "contact_email": None}
        samples = [
            ("Правильная структура и два факта", json.dumps(valid)),
            ("Пояснение перед JSON", "Вот ваш JSON: " + json.dumps(valid)),
            ("Нет обязательного nullable-поля", json.dumps({"request_id": 28, "category": "access"})),
            ("Строка вместо ID", json.dumps({**valid, "request_id": "28"})),
            ("Лишний ключ", json.dumps({**valid, "is_admin": True})),
            ("Недопустимая категория", json.dumps({**valid, "category": "administrator"})),
            ("Неверный ID при правильной структуре", json.dumps({**valid, "request_id": 82})),
            ("Выдуманный контакт", json.dumps({**valid, "contact_email": "fictional@example.test"})),
            ("Допустимая, но неверная по смыслу категория", json.dumps({**valid, "category": "billing"})),
        ]
        for label, raw in samples:
            print("\n" + label + ":")
            review_candidate(raw)
        return

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    try:
        url, key, model = resolve_provider(args.provider, args.model, os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(f"Один живой запрос: {args.provider}, {model}, режим {args.mode}. Возможны расход квоты и оплата.", flush=True)
    print("Источник синтетический. Неподдержанный режим не заменяется другим автоматически.")
    try:
        with OpenAI(base_url=url, api_key=key, timeout=30.0, max_retries=0) as client:
            response = request_completion(client, model, args.mode, max_tokens=args.max_tokens)
    except APITimeoutError:
        raise SystemExit("Тайм-аут. Удалённое вычисление и расход не считаем автоматически нулевыми.") from None
    except APIConnectionError:
        raise SystemExit("Нет соединения: проверь сеть или Ollama. Облачного fallback нет.") from None
    except APIStatusError as error:
        raise SystemExit(f"API вернул HTTP {error.status_code}. Проверь модель, доступ и поддержку выбранного response_format.") from None
    try:
        raw = extract_json_text(response)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if not review_candidate(raw):
        raise SystemExit(1)
    print("Успех локальных проверок не равен разрешению создать или изменить заявку.")


if __name__ == "__main__":
    main()
