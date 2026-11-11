"""Урок 1.3 · Рассматриваем токены, ID и байты в двух словарях tiktoken.

Запуск:
    uv run python lessons/01-03-tokenization/practice/01_inspect_tokens.py
    uv run python lessons/01-03-tokenization/practice/01_inspect_tokens.py --text "Свой текст"

Это НЕ точный счётчик Gemini. API-ключи и генерация не используются.
При первом запуске библиотека может скачать файлы словарей в кэш.
"""

import argparse

import tiktoken


def split_tokens(text: str, encoding: tiktoken.Encoding) -> list[tuple[int, bytes]]:
    """Разбираем обычный текст на ID и байты; видимые спецмаркеры не считаем командами."""
    ids = encoding.encode_ordinary(text)
    return [(token_id, encoding.decode_single_token_bytes(token_id)) for token_id in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Токены текста: ID, байты, восстановление")
    parser.add_argument("--text", help="Свой текст вместо набора примеров")
    parser.add_argument("--encoding", choices=["cl100k_base", "o200k_base"], default="o200k_base")
    args = parser.parse_args()
    encoding = tiktoken.get_encoding(args.encoding)

    samples = [args.text] if args.text is not None else ["Hello, world!", "Привет, мир!", "👩‍💻"]
    print(f"Словарь: {encoding.name}. Это tiktoken, НЕ токенизатор Gemini.\n")
    for text in samples:
        pieces = split_tokens(text, encoding)
        print(f"Текст: {text!r}")
        print(f"  Кодовых точек Python: {len(text)}, байтов UTF-8: {len(text.encode('utf-8'))}, токенов: {len(pieces)}")
        print("  ID       | фрагмент (или экранированные байты) | байты UTF-8 в hex")
        for token_id, raw in pieces:
            # Часть многобайтового символа не обязана быть самостоятельной UTF-8 строкой.
            display = raw.decode("utf-8", errors="backslashreplace")
            print(f"  {token_id:>8} | {display!r:35} | {raw.hex(' ')}")
        restored = encoding.decode([token_id for token_id, _ in pieces])
        print(f"  Полная последовательность восстановилась: {restored == text}\n")

    print("Одинаковые примеры, разные словари (считаем текст, а не сообщения API):")
    comparisons = [
        "The cat is sleeping on the couch.",
        "Кошка спит на диване.",
        "def add(a, b):\n    return a + b",
        "👩‍💻",
    ]
    encodings = [tiktoken.get_encoding(name) for name in ["cl100k_base", "o200k_base"]]
    print(f"  {'cl100k':>7} {'o200k':>7}  текст")
    for text in comparisons:
        sizes = [len(enc.encode_ordinary(text)) for enc in encodings]
        print(f"  {sizes[0]:>7} {sizes[1]:>7}  {text!r}")
    print("\nНельзя переносить эти числа на Gemini. Его usage включает собственную токенизацию и оформление запроса.")


if __name__ == "__main__":
    main()
