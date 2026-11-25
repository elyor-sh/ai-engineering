"""Локальные проверки выбора токенов и согласованности игрушечного KV-cache."""

import runpy
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "01_sampling.py"
        assert source.is_file(), "Опыт с выбором токенов ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def setUp(self):
        self.logits = np.log([0.60, 0.25, 0.10, 0.05])

    def test_temperature_one_recovers_original_distribution(self):
        np.testing.assert_allclose(self.experiment["token_distribution"](self.logits), [0.60, 0.25, 0.10, 0.05])

    def test_zero_temperature_uses_greedy_with_explicit_tie_rule(self):
        np.testing.assert_array_equal(self.experiment["token_distribution"](np.array([2.0, 2.0, 1.0]), temperature=0), [1, 0, 0])

    def test_lower_temperature_sharpens_leading_candidate(self):
        distribution = self.experiment["token_distribution"]
        self.assertGreater(distribution(self.logits, temperature=0.5)[0], distribution(self.logits)[0])

    def test_top_p_keeps_candidate_crossing_threshold(self):
        result = self.experiment["token_distribution"](self.logits, top_p=0.8)
        np.testing.assert_allclose(result, [0.60 / 0.85, 0.25 / 0.85, 0, 0])

    def test_exact_mass_threshold_needs_no_extra_candidate(self):
        np.testing.assert_allclose(self.experiment["token_distribution"](self.logits, top_p=0.6), [1, 0, 0, 0])

    def test_top_p_one_does_not_filter_probability_mass(self):
        result = self.experiment["token_distribution"](self.logits, top_p=1.0)
        self.assertTrue(np.all(result > 0))
        self.assertAlmostEqual(result.sum(), 1)

    def test_top_p_one_preserves_even_tiny_positive_tail(self):
        logits = np.log([1.0 - 1e-15, 1e-15, 1e-30])
        result = self.experiment["token_distribution"](logits, top_p=1.0)
        self.assertGreater(result[2], 0)

    def test_large_logits_have_finite_distribution(self):
        result = self.experiment["token_distribution"](np.array([1000.0, 999.0]))
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result.sum(), 1)

    def test_invalid_settings_and_nonfinite_logits_are_rejected(self):
        distribution = self.experiment["token_distribution"]
        for kwargs in [{"temperature": -1}, {"top_p": 0}, {"top_p": 1.1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                distribution(self.logits, **kwargs)
        with self.assertRaises(ValueError):
            distribution(np.array([np.nan, 0.0]))


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "02_kv_cache.py"
        assert source.is_file(), "Опыт с KV-cache ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(7, 3))
        self.wq = rng.normal(size=(3, 2))
        self.wk = rng.normal(size=(3, 2))
        self.wv = rng.normal(size=(3, 4))

    def test_prefill_with_equal_scores_averages_values(self):
        output, _ = self.experiment["prefill"](np.array([[1.0, 0.0], [0.0, 2.0]]), np.zeros((2, 2)), np.eye(2), np.eye(2))
        np.testing.assert_allclose(output, [[0.5, 1.0]])

    def test_cached_steps_match_fresh_prefix_computation(self):
        prefill, decode = self.experiment["prefill"], self.experiment["decode_step"]
        _, cache = prefill(self.x[:2], self.wq, self.wk, self.wv)
        for i in range(2, len(self.x)):
            output, cache = decode(self.x[i : i + 1], self.wq, self.wk, self.wv, cache)
            expected, _ = prefill(self.x[: i + 1], self.wq, self.wk, self.wv)
            np.testing.assert_allclose(output, expected, atol=1e-12)

    def test_cache_adds_one_position_and_keeps_correct_dimensions(self):
        _, cache = self.experiment["prefill"](self.x[:2], self.wq, self.wk, self.wv)
        _, new_cache = self.experiment["decode_step"](self.x[2:3], self.wq, self.wk, self.wv, cache)
        self.assertEqual(new_cache[0].shape, (3, 2))
        self.assertEqual(new_cache[1].shape, (3, 4))

    def test_decoding_does_not_mutate_previous_cache(self):
        _, cache = self.experiment["prefill"](self.x[:2], self.wq, self.wk, self.wv)
        before = tuple(matrix.copy() for matrix in cache)
        self.experiment["decode_step"](self.x[2:3], self.wq, self.wk, self.wv, cache)
        for old, current in zip(before, cache, strict=True):
            np.testing.assert_array_equal(current, old)

    def test_stale_cache_after_prefix_edit_is_not_a_valid_replacement(self):
        _, cache = self.experiment["prefill"](self.x[:2], self.wq, self.wk, self.wv)
        changed = self.x.copy()
        changed[0] += [5.0, -3.0, 2.0]
        stale, _ = self.experiment["decode_step"](changed[2:3], self.wq, self.wk, self.wv, cache)
        correct, _ = self.experiment["prefill"](changed[:3], self.wq, self.wk, self.wv)
        self.assertFalse(np.allclose(stale, correct))

    def test_empty_prefill_and_multirow_decode_are_rejected(self):
        with self.assertRaises(ValueError):
            self.experiment["prefill"](self.x[:0], self.wq, self.wk, self.wv)
        _, cache = self.experiment["prefill"](self.x[:2], self.wq, self.wk, self.wv)
        with self.assertRaises(ValueError):
            self.experiment["decode_step"](self.x[2:4], self.wq, self.wk, self.wv, cache)

    def test_reference_memory_estimate_is_one_gibibyte(self):
        size = self.experiment["estimate_kv_bytes"](32, 8, 128, 8192)
        self.assertEqual(size, 1024**3)

    def test_estimated_memory_scales_with_tokens_and_batch(self):
        estimate = self.experiment["estimate_kv_bytes"]
        base = estimate(32, 8, 128, 1024)
        self.assertEqual(estimate(32, 8, 128, 2048), base * 2)
        self.assertEqual(estimate(32, 8, 128, 1024, batch_size=2), base * 2)


if __name__ == "__main__":
    unittest.main()
