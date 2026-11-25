"""Урок 2.3 · Поток, завершение и условная цена.

По умолчанию синтетические события, без сети и чтения .env:
    uv run python lessons/02-03-streaming-and-costs/practice/01_streaming.py

Один настоящий вопрос только явно:
    .../01_streaming.py --live --provider gemini

Это отдельная лаборатория одного хода, не полный интерактивный чат из 2.2.
"""

import argparse
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam


@dataclass(frozen=True)
class StreamResult:
    text: str
    finish_reason: str | None
    usage: CompletionUsage | None
    status: Literal["complete", "incomplete", "cancelled", "error", "unsupported"]
    transport_complete: bool
    error_kind: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "complete"


def collect_stream(chunks: Iterable[ChatCompletionChunk], *, on_text: Callable[[str], None] | None = None) -> StreamResult:
    """Потребляем один вариант текста. Владельцем ресурса остаётся вызывающий код.

    Usage здесь — накопленный снимок попытки, не приращение. Читаем и после finish,
    потому что финальный отчёт может прийти отдельно с пустым choices.
    """
    parts: list[str] = []
    finish: str | None = None
    usage: CompletionUsage | None = None
    unsupported = False
    try:
        for chunk in chunks:
            if chunk.usage is not None:
                usage = chunk.usage  # заменить снимок, НЕ прибавить
            if not chunk.choices:
                continue
            if len(chunk.choices) != 1 or chunk.choices[0].index != 0:
                raise ValueError("Лаборатория поддерживает ровно один вариант с index=0")
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.tool_calls or delta.refusal:
                unsupported = True
            if delta.content:
                if finish is not None:
                    raise ValueError("Текст пришёл после объявленного завершения")
                parts.append(delta.content)  # пробелы не обрезаем
                if on_text is not None:
                    on_text(delta.content)
            if choice.finish_reason is not None:
                if finish is not None and finish != choice.finish_reason:
                    raise ValueError("Противоречивые причины завершения")
                finish = choice.finish_reason
    except KeyboardInterrupt:
        return StreamResult("".join(parts), finish, usage, "cancelled", False, "KeyboardInterrupt")
    except (APIError, OSError, ValueError) as exc:  # ожидаемые ошибки API, транспорта и протокола; не скрываем произвольные баги
        return StreamResult("".join(parts), finish, usage, "error", False, type(exc).__name__)

    text = "".join(parts)
    if unsupported or finish not in (None, "stop", "length"):
        status = "unsupported"
    elif finish == "stop" and text.strip():
        status = "complete"
    else:
        status = "incomplete"
    return StreamResult(text, finish, usage, status, True)


def estimate_cost(result: StreamResult, input_rate: Decimal | None, output_rate: Decimal | None) -> Decimal | None:
    """Условный USD-тариф за 1М: все prompt по одной ставке, все completion по другой.

    Cached/reasoning в этой схеме вложены в агрегаты и НЕ добавляются сверху.
    Это оценка по допущениям, не окончательный счёт провайдера.
    """
    for rate in (input_rate, output_rate):
        if rate is not None and (not rate.is_finite() or rate < 0):
            raise ValueError("Ставки должны быть конечными и неотрицательными")
    if input_rate is None or output_rate is None or result.usage is None:
        return None
    if not result.transport_complete or result.finish_reason is None:
        return None  # промежуточный снимок после обрыва не считаем финальным
    usage = result.usage
    if usage.prompt_tokens < 0 or usage.completion_tokens < 0:
        raise ValueError("Отрицательные счётчики токенов недопустимы")
    return (Decimal(usage.prompt_tokens) * input_rate + Decimal(usage.completion_tokens) * output_rate) / Decimal(1_000_000)


def commit_turn(history: Sequence[tuple[str, str]], user_text: str, result: StreamResult) -> list[tuple[str, str]]:
    """Черновик не попадает в список завершённых пар. Исходный список не изменяется."""
    if not user_text.strip():
        raise ValueError("Вопрос не должен быть пустым")
    return [*history, (user_text, result.text)] if result.accepted else list(history)


def stream_request(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 512,
    request_usage: bool = False,
    on_text: Callable[[str], None] | None = None,
) -> StreamResult:
    """Ровно одна попытка, один вопрос. Контекстный менеджер закрывает поток при любом исходе."""
    if not model.strip() or not prompt.strip() or max_tokens <= 0:
        raise ValueError("Нужны модель, непустой вопрос и положительный лимит")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "Ты — учебный помощник. Отвечай кратко по-русски."},
        {"role": "user", "content": prompt},
    ]
    if request_usage:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
    else:
        stream = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, stream=True)
    with stream as events:
        return collect_stream(events, on_text=on_text)


def resolve_provider(provider: str, model_override: str | None, env: Mapping[str, str]) -> tuple[str, str, str]:
    """Не печатай возвращённый кортеж: он содержит ключ."""
    if provider == "gemini":
        url, key, variable = "https://generativelanguage.googleapis.com/v1beta/openai/", env.get("GEMINI_API_KEY", ""), "MODEL_GEMINI"
        if not key.strip():
            raise ValueError("Не задан GEMINI_API_KEY")
    elif provider == "local":
        url, key, variable = "http://localhost:11434/v1", "ollama", "MODEL_LOCAL"
    else:
        raise ValueError("Провайдер должен быть gemini или local")
    model = model_override if model_override is not None else env.get(variable, "")
    if not model.strip():
        raise ValueError(f"Укажи --model или {variable}; скрытой модели по умолчанию нет")
    return url, key, model.strip()


def demo_events(finish: str | None = "stop", include_usage: bool = True) -> list[ChatCompletionChunk]:
    """Синтетические SDK-события. Количество токенов задано, не выведено из chunks."""
    events: list[dict[str, object]] = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "Привет, "}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "Алекс!"}, "finish_reason": None}]},
    ]
    if finish is not None:
        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if include_usage:
        events.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    base = {"id": "demo-response", "object": "chat.completion.chunk", "created": 0, "model": "demo-not-api"}
    return [ChatCompletionChunk.model_validate({**base, **event}) for event in events]


def show_report(result: StreamResult, input_rate: Decimal | None, output_rate: Decimal | None) -> None:
    print(f"\nСтатус: {result.status}; finish={result.finish_reason}; поток нормально закончился: {result.transport_complete}")
    print("Можно принять текстовую пару:", result.accepted)
    if result.error_kind is not None:
        print("Категория проблемы:", result.error_kind)
    if result.usage is None:
        print("Usage не получен. Это не нулевой расход.")
    else:
        print(f"Последний снимок usage: вход {result.usage.prompt_tokens}, выход {result.usage.completion_tokens}")
    cost = estimate_cost(result, input_rate, output_rate)
    if cost is None:
        print("Итоговая цена не определена: нет тарифа, отчёта или подтверждённого конца попытки.")
    else:
        print(f"Оценка по условному двухставочному тарифу: ${cost:.6f}; это не квитанция провайдера.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Потоковый потребитель: локальные сценарии или один явный API-запрос")
    parser.add_argument("--live", action="store_true", help="разрешить реальный запрос")
    parser.add_argument("--provider", choices=["gemini", "local"], default="gemini")
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="Поздоровайся с Алексом одним предложением.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--include-usage", action="store_true", help="только если endpoint поддерживает stream_options.include_usage")
    parser.add_argument("--input-rate", type=Decimal, help="USD за 1М входных в простом учебном тарифе")
    parser.add_argument("--output-rate", type=Decimal, help="USD за 1М выходных в простом учебном тарифе")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens должен быть положительным")
    if (args.input_rate is None) != (args.output_rate is None):
        parser.error("Укажи обе ставки или ни одной")
    for rate in (args.input_rate, args.output_rate):
        if rate is not None and (not rate.is_finite() or rate < 0):
            parser.error("Ставки должны быть конечными и неотрицательными")

    def display(text: str) -> None:
        print(text, end="", flush=True)

    if not args.live:
        print("Без сети: синтетические события, .env не читается. Цена $1 / $5 за 1М выдумана.\n")
        scenarios = [
            ("Полный текст и поздний usage", demo_events()),
            ("Полный текст без usage", demo_events(include_usage=False)),
            ("EOF без finish reason", demo_events(finish=None, include_usage=False)),
            ("Ограничение длины с usage", demo_events(finish="length")),
        ]
        for label, events in scenarios:
            print(label + ":")
            result = collect_stream(events, on_text=display)
            show_report(result, Decimal(1), Decimal(5))
            print("Принятых пар из одной попытки:", len(commit_turn([], "Поздоровайся с Алексом", result)), "\n")
        return

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    try:
        url, key, model = resolve_provider(args.provider, args.model, os.environ)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Живой режим: {args.provider}, модель {model}. Это может расходовать квоту или деньги.")
    print("Один вопрос, не полный чат. Отмена — Ctrl+C; тарифы не угадываются по имени модели.")
    try:
        with OpenAI(base_url=url, api_key=key, timeout=30.0, max_retries=0) as client:
            result = stream_request(
                client,
                model,
                args.prompt,
                max_tokens=args.max_tokens,
                request_usage=args.include_usage,
                on_text=display,
            )
    except APITimeoutError:
        raise SystemExit("Тайм-аут до обработки потока. Завершение и расход не подтверждены.") from None
    except APIConnectionError:
        raise SystemExit("Нет соединения. Проверь сеть или Ollama; облачного fallback нет.") from None
    except APIStatusError as exc:
        raise SystemExit(f"API вернул HTTP {exc.status_code}. Проверь модель, квоту и поддержку запрошенных опций.") from None
    except KeyboardInterrupt:
        raise SystemExit("Отменено до получения результата потока. Удалённый расход неизвестен.") from None
    show_report(result, args.input_rate, args.output_rate)
    print("Принятых пар этой попытки:", len(commit_turn([], args.prompt, result)))
    if not result.accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
