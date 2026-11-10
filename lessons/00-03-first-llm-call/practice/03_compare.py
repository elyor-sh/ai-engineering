"""Урок 0.3 · Эксперимент 3 — один вопрос, разные модели: время, токены, деньги.

Запуск:
    uv run python lessons/00-03-first-llm-call/practice/03_compare.py
    uv run python lessons/00-03-first-llm-call/practice/03_compare.py --providers local,fast
    uv run python lessons/00-03-first-llm-call/practice/03_compare.py -q "Свой вопрос"

Опрашиваются только провайдеры, для которых есть ключ в .env (Ollama — всегда).
"""

import argparse

from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from providers import Answer, ask, available_providers, explain_error, price_per_million

DEFAULT_QUESTION = "Объясни десятилетнему ребёнку, что такое нейросеть. Ровно 3 предложения."

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--question", default=DEFAULT_QUESTION)
parser.add_argument("--providers", help="через запятую: local,fast,flagship,gemini")
args = parser.parse_args()

console = Console()
providers = available_providers(args.providers.split(",") if args.providers else None)
messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": args.question}]

console.print(f"[bold]Вопрос:[/] {args.question}\n")

answers: list[Answer] = []
errors: dict[str, str] = {}
for p in providers:
    with console.status(f"Спрашиваю {p.label} ({p.model})..."):
        try:
            answers.append(ask(p, messages, max_tokens=10000))
        except Exception as e:  # noqa: BLE001 — показываем ошибку и идём к следующей модели
            errors[p.label] = explain_error(p, e)


def money(x: float | None) -> str:
    if x is None:
        return "?"
    if x == 0:
        return "$0"
    return f"${x:.6f}" if x < 0.01 else f"${x:,.2f}"


table = Table(title="Сравнение моделей", show_lines=True)
COLUMNS = [
    "Модель",
    "Время",
    "Токены\nвход → выход",
    "из них\nreasoning",
    "ток/с",
    "Цена за 1М\nвход / выход",
    "1 запрос",
    "1 млн\nзапросов",
    "finish",
]
for col in COLUMNS:
    table.add_column(col, justify="left" if col == "Модель" else "right")

for a in answers:
    price = price_per_million(a.provider)
    price_str = "?" if price is None else f"${price[0]:.2f} / ${price[1]:.2f}"
    table.add_row(
        f"[bold]{a.provider.label}[/]\n{a.provider.model}",
        f"{a.seconds:.1f} с",
        f"{a.prompt_tokens} → {a.completion_tokens}",
        str(a.reasoning_tokens or "—"),
        f"{a.tokens_per_second:.0f}",
        price_str,
        money(a.cost),
        money(a.cost * 1_000_000 if a.cost is not None else None),
        a.finish_reason or "?",
    )
console.print(table)

for label, err in errors.items():
    console.print(f"[red]❌ {label}:[/] {err}")

for a in answers:
    body = a.text or "[dim]<пустой ответ — проверь finish_reason>[/]"
    if a.reasoning:
        body += f"\n\n[dim]💭 размышления: {len(a.reasoning)} символов[/]"
    console.print(Panel(body, title=f"{a.provider.label} · {a.provider.model}", expand=False))

if not any(p.key_env for p in providers):
    console.print("\n[yellow]Облачных ключей нет — сравнение только с локальной моделью. Добавь OPENROUTER_API_KEY в .env.[/]")
