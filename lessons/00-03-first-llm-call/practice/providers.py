"""«Универсальная розетка» из урока 0.2 — в коде.

Любой OpenAI-совместимый провайдер описывается тремя настройками:
    1) адрес сервера (base_url)
    2) ключ (api_key)
    3) название модели (model)

Модели по умолчанию можно переопределить в .env, не трогая код:
    MODEL_LOCAL=qwen3.5:9b
    MODEL_FAST=deepseek/deepseek-v4.1-flash
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import openai
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv(find_dotenv(usecwd=True))  # ищет .env от текущей папки вверх → найдёт в корне проекта

OPENROUTER_URL = "https://openrouter.ai/api/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"


@dataclass(frozen=True)
class Provider:
    label: str  # имя для таблиц
    base_url: str  # 1) адрес
    key_env: str | None  # 2) имя переменной с ключом в .env (None — ключ не нужен)
    model: str  # 3) модель
    pricing_id: str | None = None  # id в каталоге OpenRouter для расчёта цены (None — бесплатно)

    @property
    def api_key(self) -> str | None:
        if self.key_env is None:
            return "ollama"  # Ollama ключ не проверяет, но SDK требует непустую строку
        return os.getenv(self.key_env) or None

    @property
    def available(self) -> bool:
        return self.api_key is not None

    def client(self, timeout: float = 180) -> OpenAI:
        if not self.available:
            raise RuntimeError(f"{self.label}: в .env не задан {self.key_env}")
        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout, max_retries=1)


# Названия моделей меняются часто. Актуальные: https://openrouter.ai/models и https://ollama.com/library
_local = os.getenv("MODEL_LOCAL", "qwen3.5:4b")
_fast = os.getenv("MODEL_FAST", "qwen/qwen3.7-flash")
_flagship = os.getenv("MODEL_FLAGSHIP", "anthropic/claude-sonnet-5.5")
_gemini = os.getenv("MODEL_GEMINI", "gemini-3.8-flash")

PROVIDERS: dict[str, Provider] = {
    "local": Provider("Ollama · локально", OLLAMA_URL, None, _local),
    "fast": Provider("OpenRouter · дешёвая", OPENROUTER_URL, "OPENROUTER_API_KEY", _fast, pricing_id=_fast),
    "flagship": Provider("OpenRouter · флагман", OPENROUTER_URL, "OPENROUTER_API_KEY", _flagship, pricing_id=_flagship),
    # Цена Gemini — по прайсу OpenRouter; на бесплатном тарифе AI Studio фактически $0
    "gemini": Provider("Gemini · AI Studio", GEMINI_URL, "GEMINI_API_KEY", _gemini, pricing_id=f"google/{_gemini}"),
}


# ─────────────────────────── цены ───────────────────────────


@cache
def _openrouter_prices() -> dict[str, tuple[float, float]]:
    """Цены за 1 токен (вход, выход) из публичного каталога OpenRouter.

    Эндпоинт /models публичный, поэтому работает и без ключа. Список моделей — часть
    OpenAI-совместимого API, а дополнительное поле pricing OpenRouter добавляет от себя.
    """
    catalog = OpenAI(base_url=OPENROUTER_URL, api_key=os.getenv("OPENROUTER_API_KEY") or "public", timeout=20)
    try:
        models = list(catalog.models.list())
    except openai.OpenAIError:
        return {}
    prices: dict[str, tuple[float, float]] = {}
    for m in models:
        p = getattr(m, "pricing", None) or {}
        try:
            prices[m.id] = (float(p["prompt"]), float(p["completion"]))
        except (KeyError, TypeError, ValueError):
            continue
    return prices


def price_per_million(provider: Provider) -> tuple[float, float] | None:
    """(вход, выход) в долларах за 1 млн токенов; (0, 0) для локальных; None — цена неизвестна."""
    if provider.pricing_id is None:
        return (0.0, 0.0)
    price = _openrouter_prices().get(provider.pricing_id)
    return None if price is None else (price[0] * 1e6, price[1] * 1e6)


# ─────────────────────────── вызов ───────────────────────────

_THINK_TAG = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


@dataclass
class Answer:
    provider: Provider
    text: str
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int  # включает reasoning_tokens
    reasoning_tokens: int
    seconds: float

    @property
    def cost(self) -> float | None:
        price = price_per_million(self.provider)
        if price is None:
            return None
        return (self.prompt_tokens * price[0] + self.completion_tokens * price[1]) / 1e6

    @property
    def tokens_per_second(self) -> float:
        # Грубая оценка: делим на полное время, включая время до первого токена
        return self.completion_tokens / self.seconds if self.seconds else 0.0


def _extract_reasoning(message: object, text: str) -> tuple[str, str | None]:
    """Разные провайдеры отдают «размышления» по-разному: отдельным полем или тегами <think>."""
    for field in ("reasoning", "reasoning_content"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return text, value.strip()
    match = _THINK_TAG.search(text)
    if match:
        return _THINK_TAG.sub("", text).strip(), match.group(1).strip()
    return text, None


def ask(provider: Provider, messages: Iterable[ChatCompletionMessageParam], **params) -> Answer:
    """Один вызов модели + замер времени + разбор ответа."""
    client = provider.client()
    start = time.perf_counter()
    response = client.chat.completions.create(model=provider.model, messages=messages, stream=False, **params)
    seconds = time.perf_counter() - start

    choice = response.choices[0]
    text, reasoning = _extract_reasoning(choice.message, choice.message.content or "")
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return Answer(
        provider=provider,
        text=text,
        reasoning=reasoning,
        finish_reason=choice.finish_reason,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        seconds=seconds,
    )


def explain_error(provider: Provider, error: Exception) -> str:
    """Переводит типичные ошибки API на человеческий язык (см. раздел «Когда что-то идёт не так»)."""
    is_local = provider.key_env is None
    match error:
        case openai.APIConnectionError() if is_local:
            return "Ollama не отвечает → запусти: open -a Ollama"
        case openai.APITimeoutError():
            return "Тайм-аут: модель отвечала слишком долго"
        case openai.APIConnectionError():
            return "Нет соединения с провайдером (интернет? VPN?)"
        case openai.AuthenticationError():
            return f"401: ключ отклонён → проверь {provider.key_env} в .env"
        case openai.NotFoundError() if is_local:
            return f"404: модель не скачана → ollama pull {provider.model}"
        case openai.NotFoundError():
            return f"404: модели «{provider.model}» нет → выбери на openrouter.ai/models"
        case openai.RateLimitError():
            return "429: слишком много запросов → подожди минуту"
        case openai.APIStatusError(status_code=402):
            return "402: закончился баланс → пополни Credits"
        case openai.APIStatusError(status_code=code) if code >= 500:
            return f"{code}: сбой на стороне провайдера → повтори позже"
        case openai.APIStatusError(status_code=code):
            return f"{code}: {error}"
        case RuntimeError():
            return str(error)
        case _:
            return f"{type(error).__name__}: {error}"


def available_providers(names: list[str] | None = None) -> list[Provider]:
    selected = names or list(PROVIDERS)
    unknown = [n for n in selected if n not in PROVIDERS]
    if unknown:
        raise SystemExit(f"Неизвестные провайдеры: {unknown}. Есть: {list(PROVIDERS)}")
    return [PROVIDERS[n] for n in selected if PROVIDERS[n].available]
