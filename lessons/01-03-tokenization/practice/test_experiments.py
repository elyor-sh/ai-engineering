"""Локальные проверки токенизации, учебного BPE и арифметики бюджета."""

import runpy
import unittest
from pathlib import Path

import tiktoken

HERE = Path(__file__).parent


class TokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "01_inspect_tokens.py"
        assert source.is_file(), "Эксперимент с токенами ещё не реализован"
        cls.experiment = runpy.run_path(str(source))
        cls.encoding = tiktoken.get_encoding("o200k_base")

    def test_round_trip_preserves_unicode_text(self):
        for text in ["Hello, world!", "Привет, мир!", "👩‍💻", "é и e\u0301", "\n  код\t"]:
            with self.subTest(text=text):
                pieces = self.experiment["split_tokens"](text, self.encoding)
                ids = [token_id for token_id, _ in pieces]
                self.assertEqual(self.encoding.decode(ids), text)
                self.assertEqual(b"".join(raw for _, raw in pieces), text.encode("utf-8"))

    def test_empty_text_has_no_tokens(self):
        self.assertEqual(self.experiment["split_tokens"]("", self.encoding), [])

    def test_special_marker_in_user_text_is_encoded_as_ordinary_text(self):
        text = "Пользователь написал <|endoftext|>"
        pieces = self.experiment["split_tokens"](text, self.encoding)
        ids = [token_id for token_id, _ in pieces]
        self.assertEqual(self.encoding.decode(ids), text)
        self.assertNotIn(self.encoding.eot_token, ids)

    def test_each_encoding_uses_its_own_token_ids(self):
        for name in ["cl100k_base", "o200k_base"]:
            encoding = tiktoken.get_encoding(name)
            pieces = self.experiment["split_tokens"]("Кот спит на диване.", encoding)
            self.assertEqual([token_id for token_id, _ in pieces], encoding.encode_ordinary("Кот спит на диване."))


class BpeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "02_toy_bpe.py"
        assert source.is_file(), "Учебный BPE ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_overlapping_pairs_are_merged_without_reusing_symbols(self):
        self.assertEqual(self.experiment["merge_pair"](("a", "a", "a", "a"), ("a", "a")), ("aa", "aa"))

    def test_training_learns_a_frequent_pair(self):
        merges = self.experiment["learn_merges"](["ab", "ab"], steps=1)
        self.assertEqual(merges, [("a", "b")])
        self.assertEqual(self.experiment["encode_word"]("ab", merges), ("ab", "</w>"))

    def test_encoding_unseen_words_preserves_characters_and_rules(self):
        merges = self.experiment["learn_merges"](["кот", "кот", "котик"], steps=5)
        before = merges.copy()
        for word in ["котёнок", "пёс", ""]:
            encoded = self.experiment["encode_word"](word, merges)
            self.assertEqual("".join(encoded).removesuffix("</w>"), word)
        self.assertEqual(merges, before)

    def test_zero_merges_leaves_characters(self):
        merges = self.experiment["learn_merges"](["кот"], steps=0)
        self.assertEqual(merges, [])
        self.assertEqual(self.experiment["encode_word"]("кот", merges), ("к", "о", "т", "</w>"))

    def test_empty_training_set_has_no_merges(self):
        self.assertEqual(self.experiment["learn_merges"]([], steps=10), [])


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "03_context_budget.py"
        assert source.is_file(), "Расчёт бюджета ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_document_budget_matches_lesson(self):
        self.assertEqual(self.experiment["remaining_context"](8192, 2200, 1000, margin=300), 4692)

    def test_overflow_and_exact_fit(self):
        remaining = self.experiment["remaining_context"]
        self.assertEqual(remaining(8192, 7100, 1000, margin=300), -208)
        self.assertEqual(remaining(1000, 700, 200, margin=100), 0)

    def test_price_uses_actual_output_tokens(self):
        price = self.experiment["cost_usd"](5600, 1000, 1.0, 5.0)
        self.assertAlmostEqual(price, 0.0106)
        self.assertAlmostEqual(price * 10_000, 106)

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            self.experiment["remaining_context"](1000, -1, 200)
        with self.assertRaises(ValueError):
            self.experiment["cost_usd"](10, -1, 1.0, 5.0)


if __name__ == "__main__":
    unittest.main()
