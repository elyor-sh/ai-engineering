"""Проверки вычисления одной головы attention. Сервисы и обученные модели не нужны."""

import runpy
import unittest
from pathlib import Path

import numpy as np


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name("attention_core.py")
        assert source.is_file(), "Вычисление attention ещё не реализовано"
        cls.core = runpy.run_path(str(source))

    def test_softmax_rows_sum_to_one(self):
        result = self.core["softmax_rows"](np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
        np.testing.assert_allclose(result.sum(axis=1), [1, 1])
        self.assertTrue(np.all(result >= 0))

    def test_softmax_is_stable_for_large_logits(self):
        result = self.core["softmax_rows"](np.array([[1000.0, 1001.0]]))
        np.testing.assert_allclose(result, [[0.2689414214, 0.7310585786]])

    def test_masked_score_has_zero_weight(self):
        result = self.core["softmax_rows"](np.array([[0.0, -np.inf]]))
        np.testing.assert_array_equal(result, [[1, 0]])

    def test_fully_masked_row_is_rejected(self):
        with self.assertRaises(ValueError):
            self.core["softmax_rows"](np.array([[-np.inf, -np.inf]]))

    def test_nan_and_positive_infinity_are_rejected(self):
        for value in [np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.core["softmax_rows"](np.array([[value, 0.0]]))

    def test_equal_scores_average_values_instead_of_sampling(self):
        output, weights = self.core["attention"](np.zeros((1, 2)), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[10.0, 0.0], [0.0, 20.0]]))
        np.testing.assert_allclose(weights, [[0.5, 0.5]])
        np.testing.assert_allclose(output, [[5, 10]])

    def test_scores_are_scaled_by_square_root_of_key_dimension(self):
        _, weights = self.core["attention"](
            np.array([[2.0, 0.0, 0.0, 0.0]]),
            np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
            np.eye(2),
        )
        np.testing.assert_allclose(weights, [[0.880797078, 0.119202922]])

    def test_causal_mask_excludes_future_and_normalizes_each_row(self):
        x = np.eye(4)
        _, weights = self.core["attention"](x, x, x, causal=True)
        np.testing.assert_array_equal(weights[np.triu_indices(4, k=1)], np.zeros(6))
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(4))
        self.assertEqual(weights[0, 0], 1)

    def test_future_input_does_not_change_earlier_causal_outputs(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        changed = x.copy()
        changed[-1] = [4.0, -3.0]
        before, _ = self.core["attention"](x, x, x, causal=True)
        after, _ = self.core["attention"](changed, changed, changed, causal=True)
        np.testing.assert_allclose(before[:-1], after[:-1])

    def test_future_input_can_change_earlier_unmasked_outputs(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        changed = x.copy()
        changed[-1] = [4.0, -3.0]
        before, _ = self.core["attention"](x, x, x)
        after, _ = self.core["attention"](changed, changed, changed)
        self.assertFalse(np.allclose(before[:-1], after[:-1]))

    def test_unmasked_attention_without_positions_is_permutation_equivariant(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        order = [2, 0, 1]
        before, _ = self.core["attention"](x, x, x)
        permuted, _ = self.core["attention"](x[order], x[order], x[order])
        np.testing.assert_allclose(permuted, before[order])

    def test_value_dimension_can_differ_from_key_dimension(self):
        output, _ = self.core["attention"](np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 5)))
        self.assertEqual(output.shape, (2, 5))

    def test_incompatible_shapes_are_rejected(self):
        attention = self.core["attention"]
        cases = [
            (np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 5)), False),
            (np.ones((2, 3)), np.ones((2, 3)), np.ones((3, 5)), False),
            (np.ones((1, 3)), np.ones((2, 3)), np.ones((2, 5)), True),
        ]
        for q, k, v, causal in cases:
            with self.subTest(shapes=(q.shape, k.shape, v.shape), causal=causal), self.assertRaises(ValueError):
                attention(q, k, v, causal=causal)

    def test_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.core["attention"](np.array([[np.nan, 1.0]]), np.ones((1, 2)), np.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
