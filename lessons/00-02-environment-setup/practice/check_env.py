#!/usr/bin/env python3
"""Проверка рабочего окружения курса AI Engineering.

Использует только стандартную библиотеку — работает даже до установки пакетов.

Запуск из корня репозитория:
    uv run python lessons/00-02-environment-setup/practice/check_env.py
    uv run python lessons/00-02-environment-setup/practice/check_env.py --ping

--ping дополнительно проверяет, что API-ключи рабочие. Это бесплатно:
скрипт только запрашивает список моделей / информацию о ключе, без генерации.
"""

from __future__ import annotations

import argparse
import http.client
import importlib.metadata as md
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.parse
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MIN_PYTHON = (3, 12)

REQUIRED_PACKAGES = ["openai", "anthropic", "python-dotenv", "rich"]
DEV_PACKAGES = ["ipykernel", "jupyterlab"]
RECOMMENDED_MODELS = ["qwen3.5:4b", "qwen3-embedding:0.6b"]

# env-переменная -> (название, URL для бесплатной проверки, функция построения заголовков)
PROVIDERS: dict[str, tuple[str, str, Callable[[str], dict[str, str]]]] = {
    "OPENROUTER_API_KEY": (
        "OpenRouter",
        "https://openrouter.ai/api/v1/key",
        lambda k: {"Authorization": f"Bearer {k}"},
    ),
    "GEMINI_API_KEY": (
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/models",
        lambda k: {"x-goog-api-key": k},
    ),
    "OPENAI_API_KEY": (
        "OpenAI",
        "https://api.openai.com/v1/models",
        lambda k: {"Authorization": f"Bearer {k}"},
    ),
    "ANTHROPIC_API_KEY": (
        "Anthropic",
        "https://api.anthropic.com/v1/models",
        lambda k: {"x-api-key": k, "anthropic-version": "2023-06-01"},
    ),
}


class Report:
    """Собирает результаты проверок и печатает их."""

    def __init__(self) -> None:
        self.fails = 0
        self.warns = 0

    def section(self, title: str) -> None:
        print(f"\n── {title} " + "─" * max(0, 50 - len(title)))

    def ok(self, msg: str) -> None:
        print(f"  ✅ {msg}")

    def warn(self, msg: str, hint: str = "") -> None:
        self.warns += 1
        print(f"  ⚠️  {msg}")
        if hint:
            print(f"      → {hint}")

    def fail(self, msg: str, hint: str = "") -> None:
        self.fails += 1
        print(f"  ❌ {msg}")
        if hint:
            print(f"      → {hint}")


def ssl_context() -> ssl.SSLContext:
    """SSL-контекст: certifi, если установлен (идёт вместе с openai), иначе системный."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 10) -> tuple[int | None, str]:
    """GET-запрос. Возвращает (HTTP-статус, тело) или (None, текст ошибки сети)."""
    # http.client умеет только http/https (в отличие от urllib, который откроет и file://).
    # Это важно: URL может прийти из .env (OLLAMA_BASE_URL).
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.netloc, timeout=timeout, context=ssl_context()
        )
    elif parts.scheme == "http":
        conn = http.client.HTTPConnection(parts.netloc, timeout=timeout)
    else:
        return None, f"недопустимая схема URL: {url}"

    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException) as e:  # сеть, DNS, таймаут, SSL
        return None, str(e) or type(e).__name__
    finally:
        conn.close()


def load_dotenv_file(path: Path) -> list[str]:
    """Мини-парсер .env (без python-dotenv). Возвращает список найденных ключей."""
    found: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key, value)
            found.append(key)
    return found


# ─────────────────────────── проверки ───────────────────────────


def check_python(r: Report) -> None:
    r.section("Python")
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        r.ok(f"Python {version}")
    else:
        r.fail(f"Python {version} — нужен {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+", "uv python pin 3.12 && uv sync")

    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        r.ok(f"Виртуальное окружение: {Path(sys.prefix).relative_to(ROOT) if Path(sys.prefix).is_relative_to(ROOT) else sys.prefix}")
    else:
        r.fail("Скрипт запущен НЕ в виртуальном окружении", "Запускай через `uv run python ...`")


def check_packages(r: Report) -> None:
    r.section("Пакеты")
    for name in REQUIRED_PACKAGES:
        try:
            r.ok(f"{name} {md.version(name)}")
        except md.PackageNotFoundError:
            r.fail(f"{name} не установлен", f"uv add {name}")
    for name in DEV_PACKAGES:
        try:
            r.ok(f"{name} {md.version(name)} (dev)")
        except md.PackageNotFoundError:
            r.warn(f"{name} не установлен (нужен для Jupyter)", f"uv add --dev {name}")


def check_tools(r: Report) -> None:
    r.section("Инструменты")
    for tool, required in [("uv", True), ("git", True), ("ollama", True), ("docker", False)]:
        path = shutil.which(tool)
        if path:
            r.ok(f"{tool}: {path}")
        elif required:
            r.fail(f"{tool} не найден", "см. practice/README.md")
        else:
            r.warn(f"{tool} не найден (понадобится в модуле 5)")


def check_files_and_git(r: Report) -> None:
    r.section("Файлы проекта и git")
    for name, hint in [
        ("pyproject.toml", "uv init --bare --python 3.12"),
        ("uv.lock", "uv add ... создаст его автоматически"),
        (".python-version", "uv python pin 3.12"),
        (".gitignore", "файл должен быть в корне репозитория"),
    ]:
        if (ROOT / name).exists():
            r.ok(name)
        else:
            r.fail(f"{name} отсутствует", hint)

    if not (ROOT / ".git").exists():
        r.fail("Git-репозиторий не инициализирован", "git init")
        return
    r.ok("git-репозиторий")

    # Главная проверка безопасности: .env должен игнорироваться
    res = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT, capture_output=True)
    if res.returncode == 0:
        r.ok(".env игнорируется git — секреты не утекут")
    else:
        r.fail(".env НЕ игнорируется git!", "добавь строку `.env` в .gitignore")

    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"], cwd=ROOT, capture_output=True)
    if tracked.returncode == 0:
        r.fail(".env уже закоммичен в git!", "git rm --cached .env && перевыпусти все ключи")


def check_env_file(r: Report) -> list[str]:
    r.section("Ключи (.env)")
    env_path = ROOT / ".env"
    if not env_path.exists():
        r.fail(".env не найден", "cp .env.example .env и заполни ключи")
        return []

    load_dotenv_file(env_path)
    configured = [k for k in PROVIDERS if os.environ.get(k)]
    for key, (name, *_rest) in PROVIDERS.items():
        if key in configured:
            # Значение ключа НЕ печатаем — только факт наличия и длину
            r.ok(f"{name}: задан ({len(os.environ[key])} символов)")
        else:
            print(f"  ·  {name}: не задан")

    if not configured:
        r.warn("Нет ни одного облачного ключа", "Для курса достаточно OpenRouter или Gemini (бесплатно). Пока можно учиться на Ollama.")
    return configured


def check_ollama(r: Report) -> None:
    r.section("Ollama (локальные модели)")
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    status, body = http_get(f"{base}/api/tags", timeout=3)
    if status != 200:
        r.fail(f"Ollama не отвечает на {base}", "Запусти приложение: `open -a Ollama`")
        return
    r.ok(f"Ollama работает на {base}")

    try:
        models = {m["name"] for m in json.loads(body).get("models", [])}
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as e:
        r.fail(f"Ollama вернул неожиданный ответ: {e}", "обнови Ollama или проверь OLLAMA_BASE_URL")
        return
    for m in sorted(models):
        print(f"  ·  установлена: {m}")
    for want in RECOMMENDED_MODELS:
        if want in models or f"{want}:latest" in models:
            r.ok(f"рекомендуемая модель {want}")
        else:
            r.warn(f"нет модели {want}", f"ollama pull {want}")


def ping_providers(r: Report, configured: list[str]) -> None:
    r.section("Проверка ключей (--ping, бесплатно)")
    if not configured:
        print("  ·  нечего проверять — ключи не заданы")
        return
    for key in configured:
        name, url, make_headers = PROVIDERS[key]
        status, body = http_get(url, make_headers(os.environ[key]))
        if status == 200:
            r.ok(f"{name}: ключ рабочий")
        elif status in (400, 401, 403):
            r.fail(f"{name}: ключ отклонён (HTTP {status})", "проверь, что скопировал ключ целиком, без пробелов")
        elif status is None:
            r.warn(f"{name}: нет соединения — {body}")
        else:
            r.warn(f"{name}: неожиданный ответ HTTP {status}", body[:200])


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения курса AI Engineering")
    parser.add_argument("--ping", action="store_true", help="проверить API-ключи запросом к провайдерам")
    args = parser.parse_args()

    print(f"Корень проекта: {ROOT}")
    r = Report()
    check_python(r)
    check_packages(r)
    check_tools(r)
    check_files_and_git(r)
    configured = check_env_file(r)
    check_ollama(r)
    if args.ping:
        ping_providers(r, configured)

    print("\n" + "═" * 54)
    if r.fails == 0:
        print(f"🎉 Окружение готово! (предупреждений: {r.warns})")
        if not args.ping:
            print("   Совет: запусти с --ping, чтобы проверить API-ключи.")
    else:
        print(f"Ошибок: {r.fails}, предупреждений: {r.warns}. Исправь ❌ и запусти снова.")
    return 1 if r.fails else 0


if __name__ == "__main__":
    sys.exit(main())
