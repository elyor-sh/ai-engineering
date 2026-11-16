"""Локальные проверки целевых токенов, маски loss и игрушечной парной цели DPO."""

import runpy
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "01_training_targets.py"
        assert source.is_file(), "Опыт с целями обучения ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_targets_and_loss_mask_are_shifted_together(self):
        inputs, targets, mask = self.experiment["shift_targets"]([0, 1, 2, 3, 4], [False, False, False, True, True])
        np.testing.assert_array_equal(inputs, [0, 1, 2, 3])
        np.testing.assert_array_equal(targets, [1, 2, 3, 4])
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_short_or_misaligned_sequences_are_rejected(self):
        for ids, roles in [([0], [False]), ([0, 1], [True])]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.experiment["shift_targets"](ids, roles)

    def test_uniform_predictions_have_log_vocabulary_loss(self):
        loss = self.experiment["masked_cross_entropy"](np.zeros((2, 6)), np.array([1, 4]), np.array([True, True]))
        self.assertAlmostEqual(loss, np.log(6))

    def test_masked_prompt_predictions_do_not_change_reported_sft_loss(self):
        logits = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
        changed = logits.copy()
        changed[0] = [1000, -1000, 0]
        loss = self.experiment["masked_cross_entropy"]
        targets, mask = np.array([1, 1]), np.array([False, True])
        self.assertAlmostEqual(loss(logits, targets, mask), loss(changed, targets, mask))

    def test_worse_prediction_of_assistant_token_increases_loss(self):
        loss = self.experiment["masked_cross_entropy"]
        targets, mask = np.array([1]), np.array([True])
        self.assertLess(loss(np.array([[0.0, 3.0]]), targets, mask), loss(np.array([[3.0, 0.0]]), targets, mask))

    def test_empty_mask_and_invalid_target_are_rejected(self):
        loss = self.experiment["masked_cross_entropy"]
        with self.assertRaises(ValueError):
            loss(np.zeros((1, 2)), np.array([1]), np.array([False]))
        with self.assertRaises(ValueError):
            loss(np.zeros((1, 2)), np.array([2]), np.array([True]))


class PreferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "02_preferences_and_reward.py"
        assert source.is_file(), "Опыт с предпочтениями ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_two_answer_probabilities_sum_to_one_without_overflow(self):
        for theta in [-1000, 0, 1000]:
            with self.subTest(theta=theta):
                logs = self.experiment["answer_logprobs"](theta)
                self.assertTrue(np.isfinite(logs).all())
                self.assertAlmostEqual(np.exp(logs).sum(), 1)

    def test_policy_equal_to_reference_starts_at_log_two_loss(self):
        logs = self.experiment["answer_logprobs"](2.5)
        for chosen in [0, 1]:
            loss = self.experiment["dpo_pair_loss"](logs, logs, chosen=chosen, beta=0.5)
            self.assertAlmostEqual(loss, np.log(2))

    def test_preference_gain_is_measured_relative_to_reference(self):
        logprobs = self.experiment["answer_logprobs"]
        loss = self.experiment["dpo_pair_loss"]
        self.assertLess(loss(logprobs(2), logprobs(0), chosen=0), loss(logprobs(0), logprobs(0), chosen=0))
        self.assertAlmostEqual(loss(logprobs(2), logprobs(2), chosen=0), np.log(2))

    def test_one_update_matches_binary_dpo_gradient(self):
        theta, probabilities, losses = self.experiment["train_toy_policy"](chosen=0, steps=1, lr=0.4, beta=0.5)
        self.assertAlmostEqual(theta, 0.1)
        self.assertAlmostEqual(probabilities[-1], 0.5249791875)
        self.assertLess(losses[-1], losses[0])

    def test_correct_preference_improves_selection_probability(self):
        _, probabilities, losses = self.experiment["train_toy_policy"](chosen=0)
        self.assertGreater(probabilities[-1], probabilities[0])
        self.assertLess(losses[-1], losses[0])

    def test_wrong_preference_reduces_loss_but_hurts_correct_answer(self):
        _, probabilities, losses = self.experiment["train_toy_policy"](chosen=1)
        self.assertLess(probabilities[-1], probabilities[0])
        self.assertLess(losses[-1], losses[0])

    def test_large_unfavorable_margin_has_finite_loss(self):
        logprobs = self.experiment["answer_logprobs"]
        loss = self.experiment["dpo_pair_loss"](logprobs(-1000), logprobs(0), chosen=0)
        self.assertTrue(np.isfinite(loss))

    def test_invalid_preference_and_beta_are_rejected(self):
        logs = self.experiment["answer_logprobs"](0)
        with self.assertRaises(ValueError):
            self.experiment["dpo_pair_loss"](logs, logs, chosen=2)
        with self.assertRaises(ValueError):
            self.experiment["dpo_pair_loss"](logs, logs, chosen=0, beta=0)


if __name__ == "__main__":
    unittest.main()
