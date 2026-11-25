"""Урок 1.7 · KV-cache: результат как при пересчёте, меньше повторных K/V-проекций.

Запуск:
    uv run python lessons/01-07-text-generation/practice/02_kv_cache.py

Одна искусственная голова, фиксированные входы и веса. Нет настоящей LLM, RoPE,
MLP, батчинга или генерации токенов. Счётчики — не бенчмарк времени сервера.
"""

from pathlib import Path

import numpy as np

KVCache = tuple[np.ndarray, np.ndarray]


def _attend_last(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Запрос последней позиции видит весь переданный префикс, включая себя."""
    for matrix in (query, keys, values):
        if matrix.ndim != 2 or 0 in matrix.shape or not np.isfinite(matrix).all():
            raise ValueError("Нужны непустые матрицы конечных чисел")
    if query.shape[0] != 1 or query.shape[1] != keys.shape[1] or keys.shape[0] != values.shape[0]:
        raise ValueError("Размерности последнего Query и кэша не согласованы")
    scores = query @ keys.T / np.sqrt(keys.shape[1])
    exponents = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = exponents / exponents.sum(axis=1, keepdims=True)
    return weights @ values


def prefill(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray) -> tuple[np.ndarray, KVCache]:
    """Проецируем известный префикс. Возвращаем результат его последней позиции и K/V."""
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Prefill ожидает непустую последовательность входных векторов")
    query = x[-1:] @ wq
    keys, values = x @ wk, x @ wv
    return _attend_last(query, keys, values), (keys, values)


def decode_step(x_new: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray, cache: KVCache) -> tuple[np.ndarray, KVCache]:
    """Обрабатываем ровно одну новую позицию. Старые массивы не изменяем.

    Будущих позиций здесь нет: все ключи относятся к прошлому или текущему токену.
    Поэтому маска с локальной строкой 0 не нужна и могла бы ошибочно скрыть прошлое.
    """
    if x_new.ndim != 2 or x_new.shape[0] != 1:
        raise ValueError("Decode ожидает ровно один новый входной вектор")
    old_keys, old_values = cache
    if old_keys.shape[0] != old_values.shape[0]:
        raise ValueError("Длины K и V в кэше не совпадают")
    keys = np.concatenate([old_keys, x_new @ wk], axis=0)
    values = np.concatenate([old_values, x_new @ wv], axis=0)
    query = x_new @ wq
    return _attend_last(query, keys, values), (keys, values)


def estimate_kv_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, *, bytes_per_element: int = 2, batch_size: int = 1) -> int:
    """Только массивы K/V; предполагаем одинаковую размерность ключей и значений."""
    if min(layers, kv_heads, head_dim, bytes_per_element, batch_size) <= 0 or tokens < 0:
        raise ValueError("Размеры положительные, число токенов неотрицательное")
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per_element * batch_size


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 3))
    wq, wk, wv = rng.normal(size=(3, 2)), rng.normal(size=(3, 2)), rng.normal(size=(3, 4))
    prompt_length = 4
    cached_output, initial_cache = prefill(x[:prompt_length], wq, wk, wv)
    cache = initial_cache
    lengths = np.arange(prompt_length, len(x) + 1)
    uncached_positions = np.cumsum(lengths)
    cached_positions = lengths.copy()
    errors = []

    print("Сравниваем последнюю позицию каждого префикса. Добавляем заранее заданные векторы, не генерируем текст.")
    print("Длина | макс. разница выходов | K/V-позиций без кэша | с кэшем")
    for length, without_cache, with_cache in zip(lengths, uncached_positions, cached_positions, strict=True):
        if length > prompt_length:
            cached_output, cache = decode_step(x[length - 1 : length], wq, wk, wv, cache)
        fresh_output, _ = prefill(x[:length], wq, wk, wv)
        error = np.max(np.abs(cached_output - fresh_output))
        errors.append(error)
        print(f"{length:5} | {error:20.3e} | {without_cache:19} | {with_cache:7}")
    print("Все результаты совпали в пределах 1e-12:", np.all(np.asarray(errors) < 1e-12))
    print("Счётчики относятся к каждому способу отдельно, а не ко всем вычислениям проверочного скрипта.")

    changed = x.copy()
    changed[0] += [5.0, -3.0, 2.0]
    stale_output, _ = decode_step(changed[prompt_length : prompt_length + 1], wq, wk, wv, initial_cache)
    correct_output, _ = prefill(changed[: prompt_length + 1], wq, wk, wv)
    print("\nИзменили начало, но оставили старый кэш. Совпало с правильным пересчётом:", np.allclose(stale_output, correct_output))
    print("Игрушка не хранит идентификатор префикса: совместимость должен обеспечивать вызывающий код.")

    memory = estimate_kv_bytes(32, 8, 128, 8192)
    print(f"\nУсловная модель: 32 слоя, 8 KV-голов, 128 чисел, 8192 токена, 2 байта → {memory / 1024**3:.2f} ГиБ")
    print("Это не параметры нашей игрушки и не вся память настоящей модели.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(lengths, uncached_positions, "o-", label="Пересчёт префикса")
    axes[0].plot(lengths, cached_positions, "o-", label="С KV-cache")
    axes[0].set_title("Повторные проекции K/V в игрушке")
    axes[0].set_xlabel("Текущая длина префикса")
    axes[0].set_ylabel("Всего спроецированных позиций")
    axes[0].legend()
    token_counts = [1024, 2048, 4096, 8192, 16384]
    sizes = [estimate_kv_bytes(32, 8, 128, n) / 1024**3 for n in token_counts]
    axes[1].plot(token_counts, sizes, "o-")
    axes[1].set_title("Память условной 32-слойной модели")
    axes[1].set_xlabel("Сохранённых токенов")
    axes[1].set_ylabel("Только K и V, ГиБ")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(__file__).parent / "out" / "02_kv_cache.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Графики: {path}")
    print("72 против 12 позиций — не обещание шестикратного ускорения настоящего API.")


if __name__ == "__main__":
    main()
