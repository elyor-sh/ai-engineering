"""Урок 1.1 · Эксперимент 5 (бонус) — настоящие эмбеддинги из локальной модели.

Нужно: запущенный Ollama и модель эмбеддингов
    ollama pull qwen3-embedding:0.6b

Запуск из корня репозитория:
    uv run python lessons/01-01-math-minimum/practice/05_real_embeddings.py
"""

import os

import numpy as np
import openai
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

load_dotenv(find_dotenv(usecwd=True))

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
MODEL = os.getenv("MODEL_EMBED", "qwen3-embedding:0.6b")

SENTENCES = [
    "Кошка спит на диване",
    "Котёнок дремлет на софе",
    "The cat is sleeping on the couch",
    "Собака гуляет в парке",
    "Как подать налоговую декларацию?",
    "Сроки сдачи отчётности в налоговую",
]

console = Console()
client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")  # та же «универсальная розетка» из урока 0.3

try:
    response = client.embeddings.create(model=MODEL, input=SENTENCES)
except openai.APIConnectionError:
    raise SystemExit("❌ Ollama не отвечает → open -a Ollama") from None
except openai.NotFoundError:
    raise SystemExit(f"❌ Модель не скачана → ollama pull {MODEL}") from None

E = np.array([item.embedding for item in response.data])  # матрица: фразы × измерения
console.print(f"[bold]Модель:[/] {MODEL}")
console.print(f"[bold]Матрица эмбеддингов:[/] {E.shape}  ← {E.shape[0]} фраз, по {E.shape[1]} чисел в каждой")
console.print(f"[bold]Первые 5 чисел первой фразы:[/] {E[0, :5].round(4)} …  (что они значат? никто не знает 🙂)")
console.print(f"[bold]Длины векторов:[/] {np.linalg.norm(E, axis=1).round(3)}  ← уже нормализованы?\n")

# Косинусное сходство всех со всеми: нормализуем и умножаем матрицу на саму себя
E_unit = E / np.linalg.norm(E, axis=1, keepdims=True)
S = E_unit @ E_unit.T

table = Table(title="Косинусное сходство фраз (1.0 = одно и то же)", show_lines=False)
table.add_column("")
short = [s[:18] + ("…" if len(s) > 18 else "") for s in SENTENCES]
for i, _ in enumerate(SENTENCES):
    table.add_column(f"{i + 1}", justify="right")
for i, name in enumerate(short):
    cells = []
    for j in range(len(SENTENCES)):
        value = S[i, j]
        if i == j:
            cells.append("[dim]—[/]")
        else:
            color = "green" if value >= 0.6 else ("yellow" if value >= 0.4 else "red")
            cells.append(f"[{color}]{value:.2f}[/]")
    table.add_row(f"{i + 1}. {name}", *cells)
console.print(table)

console.print("\n[bold]Ближайший сосед каждой фразы:[/]")
for i, sentence in enumerate(SENTENCES):
    row = S[i].copy()
    row[i] = -np.inf
    j = int(row.argmax())
    console.print(f"  «{sentence}» → «{SENTENCES[j]}» ({S[i, j]:.2f})")
