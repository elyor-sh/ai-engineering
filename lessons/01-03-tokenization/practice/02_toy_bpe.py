"""Урок 1.3 · Символьная BPE-игрушка: частые пары становятся крупными деталями.

Запуск:
    uv run python lessons/01-03-tokenization/practice/02_toy_bpe.py

Только стандартная библиотека. Это НЕ полноценный байтовый BPE из tiktoken:
нет предразбиения регулярными выражениями и специальных правил реальной модели.
"""

from collections import Counter
from itertools import pairwise

Pair = tuple[str, str]
Symbols = tuple[str, ...]
END = "</w>"  # учебная отметка конца слова, не служебный токен Gemini/OpenAI


def merge_pair(symbols: Symbols, pair: Pair) -> Symbols:
    """Объединяем слева направо без перекрытий: a a a a → aa aa, а не aaa."""
    result: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


def learn_merges(words: list[str], steps: int) -> list[Pair]:
    """Частоты считаем с учётом повторов слов. При равенстве выбираем пару по алфавиту."""
    corpus = [tuple(word) + (END,) for word in words]
    rules: list[Pair] = []
    for _ in range(steps):
        counts: Counter[Pair] = Counter()
        for symbols in corpus:
            counts.update(pairwise(symbols))
        if not counts:
            break
        pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
        rules.append(pair)
        corpus = [merge_pair(symbols, pair) for symbols in corpus]
    return rules


def encode_word(word: str, merges: list[Pair]) -> Symbols:
    """Используем готовые правила: новый текст не меняет словарь и порядок объединений."""
    symbols = tuple(word) + (END,)
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return symbols


def main() -> None:
    words = ["кот", "кот", "кот", "котик", "котик", "коты"]
    print("Учебные слова:", words)
    print("Начальные детали:", list("кот") + [END])
    rules = learn_merges(words, steps=6)
    print("\nОбучение токенизатора — готовим правила:")
    for step, (left, right) in enumerate(rules, start=1):
        partial = encode_word("котик", rules[:step])
        print(f"  {step}. {left!r} + {right!r} → {left + right!r}; «котик»: {partial}")

    print("\nИспользование готового токенизатора — правила больше не меняются:")
    for word in ["кот", "котик", "коты", "котёнок", "пёс"]:
        encoded = encode_word(word, rules)
        restored = "".join(encoded).removesuffix(END)
        print(f"  {word!r:12} → {encoded} → восстановлено: {restored!r}")
    print("\nРедкое слово собирается из мелких частей. В этой игрушке исходные части — буквы, не байты.")


if __name__ == "__main__":
    main()
