"""Урок 1.4 · Таблица эмбеддингов: ID выбирает строку, а не задаёт смысл числом.

Запуск из корня проекта:
    uv run python lessons/01-04-embeddings/practice/01_embedding_lookup.py

Таблица и ID придуманы вручную. Это не токенизатор и не обученная языковая модель.
"""

import numpy as np


def lookup_embeddings(table: np.ndarray, token_ids: list[int]) -> np.ndarray:
    """Берём строки в порядке ID; повтор ID возвращает тот же начальный вектор."""
    if table.ndim != 2:
        raise ValueError("Таблица должна иметь форму (словарь, размерность)")
    indices = np.asarray(token_ids, dtype=int)
    if np.any(indices < 0) or np.any(indices >= table.shape[0]):
        raise IndexError("ID должен указывать на существующую строку таблицы")
    return table[indices]


def main() -> None:
    tokens = ["лук", "салат", "тетива", "<конец>"]
    table = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.7, 0.0, 0.2],
            [-0.4, 0.6, 0.1],
            [0.0, 0.1, -0.1],
        ]
    )
    print("Учебная таблица: строки — токены, столбцы — координаты")
    for token_id, (token, row) in enumerate(zip(tokens, table, strict=True)):
        print(f"  ID {token_id}: {token:8} → {row}")
    print(f"Форма таблицы: {table.shape}; чисел: {table.size}\n")

    ids = [0, 1, 3]
    sequence = lookup_embeddings(table, ids)
    print(f"Последовательность ID: {ids}")
    print(f"Три токена → матрица формы {sequence.shape}:\n{sequence}")

    first_context = lookup_embeddings(table, [0, 1])  # «лук», «салат»
    second_context = lookup_embeddings(table, [0, 2])  # «лук», «тетива»
    print("\nНачальный вектор «лук» в двух окружениях:")
    print("  Салат:", first_context[0])
    print("  Тетива:", second_context[0])
    print("  Совпадает:", np.array_equal(first_context[0], second_context[0]))
    print(
        "\nКонтекстные представления появятся после дальнейших слоёв, не при lookup.\n"
        "В этой игрушке таких слоёв нет. Числа не выучены, смысловую близость по ним оценивать нельзя."
    )


if __name__ == "__main__":
    main()
