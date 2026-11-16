"""Урок 1.4 · Локальные эмбеддинги: геометрия, отрицания и кандидаты для поиска.

Нужно: Ollama и модель qwen3-embedding:0.6b.
Запуск:
    uv run python lessons/01-04-embeddings/practice/02_local_embeddings.py
    uv run python lessons/01-04-embeddings/practice/02_local_embeddings.py --query "Как долго ждать доставку?"

Адрес в этом примере намеренно локальный. Ключ Gemini не используется.
"""

import argparse
import os
import time

import numpy as np
from dotenv import find_dotenv, load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

DOCUMENTS = [
    "Товар можно вернуть в течение 14 дней, если сохранён чек.",
    "Для возврата покупки сохраните чек и обратитесь в магазин в течение двух недель.",
    "Возврат товара запрещён даже при наличии чека.",
    "Доставка заказа обычно занимает от двух до пяти дней.",
    "Сбросить пароль можно через ссылку в письме.",
]


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Приводим каждую строку к единичной длине, не изменяя исходный массив."""
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Ожидается матрица (число текстов, размерность)")
    if not np.isfinite(values).all():
        raise ValueError("Вектор содержит NaN или бесконечность")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("У нулевого вектора нет направления для косинуса")
    return values / norms


def cosine_scores(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Косинус вопроса с каждой строкой; это не вероятность правильности."""
    if vectors.ndim != 2 or query.ndim != 1 or vectors.shape[1] != query.shape[0]:
        raise ValueError("Размерности документов и вопроса не согласованы")
    documents_unit = normalize_rows(vectors)
    query_unit = normalize_rows(query[None, :])[0]
    return np.clip(documents_unit @ query_unit, -1.0, 1.0)


def fetch_embeddings(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    """Возвращаем строки в порядке исходных текстов, используя index ответа API."""
    response = client.embeddings.create(model=model, input=texts, encoding_format="float")
    items = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in items] != list(range(len(texts))):
        raise ValueError("Ответ API не соответствует числу и индексам входных текстов")
    return np.array([item.embedding for item in items], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Как вернуть покупку с чеком?")
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    model = os.getenv("MODEL_EMBED", "qwen3-embedding:0.6b")

    # Qwen3 Embedding использует инструкцию задачи в запросе. Документы оставляем обычным текстом.
    # Для другой модели нужно проверить её формат и префиксы, а не переносить этот вслепую.
    query_text = args.query
    if model.startswith("qwen3-embedding"):
        query_text = f"Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: {args.query}"

    print(f"Отправляю {len(DOCUMENTS)} документов и один вопрос локальной модели {model}…", flush=True)
    started = time.perf_counter()
    try:
        with OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=30.0, max_retries=0) as client:
            vectors = fetch_embeddings(client, DOCUMENTS + [query_text], model)
    except APITimeoutError:
        raise SystemExit("Тайм-аут локальной модели. Проверь Ollama и повтори после загрузки модели в память.") from None
    except APIConnectionError:
        raise SystemExit("Ollama не отвечает → open -a Ollama") from None
    except APIStatusError as exc:
        if exc.status_code == 404:
            raise SystemExit(f"Модель не найдена → ollama pull {model}") from None
        raise SystemExit(f"Ollama вернул HTTP {exc.status_code}; проверь поддержку embeddings выбранной моделью.") from None

    print(f"Получено за {time.perf_counter() - started:.1f} с. Матрица: {vectors.shape}")
    print("Длины векторов:", np.linalg.norm(vectors, axis=1).round(3))
    print("Первые пять координат первого документа:", vectors[0, :5].round(4))
    print("По этим пяти числам нельзя прочитать смысл или восстановить исходную фразу.\n")

    docs, query = vectors[:-1], vectors[-1]
    paraphrase = cosine_scores(docs[1:2], docs[0])[0]
    contradiction = cosine_scores(docs[2:3], docs[0])[0]
    print(f"Можно вернуть ↔ перефразировка: {paraphrase:.3f}")
    print(f"Можно вернуть ↔ возврат запрещён: {contradiction:.3f}")
    print("Высокая близость противоположных правил не означает, что оба правила верны.\n")

    scores = cosine_scores(docs, query)
    print(f"Вопрос: {args.query}")
    print("Кандидаты по косинусу — это поиск, а не сгенерированный ответ:")
    for index in np.argsort(-scores):
        print(f"  {scores[index]:.3f} | {DOCUMENTS[index]}")
    scaled = cosine_scores(docs * 7, query * 3)
    print("\nПоложительное масштабирование сохранило косинусы:", np.allclose(scores, scaled))
    print("Оценки не являются вероятностями. Проверь, какие кандидаты действительно отвечают на вопрос.")


if __name__ == "__main__":
    main()
