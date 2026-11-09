"""Проверки учебных экспериментов: uv run python -m unittest discover -s <practice> -v."""

import runpy
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


class NeuronTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "01_neuron_and_layers.py"
        assert source.is_file(), "Эксперимент с нейроном ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_neuron_applies_weights_bias_and_relu(self):
        neuron = self.experiment["neuron"]
        self.assertAlmostEqual(neuron([6, 5, 8], [0.1, 0.3, 0.8], bias=-5), 3.5)
        self.assertEqual(neuron([8, 1, 1], [0.1, 0.3, 0.8], bias=-5), 0)

    def test_xor_network_matches_all_four_cases(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        output = self.experiment["xor_network"](inputs)
        np.testing.assert_allclose(output, [0, 1, 1, 0])

    def test_removing_relu_breaks_xor(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        output = self.experiment["xor_network"](inputs, use_relu=False)
        np.testing.assert_allclose(output, [2, 1, 1, 0])


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "02_learning.py"
        assert source.is_file(), "Эксперимент с обучением ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def setUp(self):
        self.x = np.array([0.5, 1.0, 1.5, 2.0])
        self.y = 2 * self.x + 1

    def test_mse_does_not_cancel_opposite_errors(self):
        loss = self.experiment["mse"](np.array([2, 4]), np.array([3, 3]))
        self.assertEqual(loss, 1)

    def test_one_update_matches_the_lesson(self):
        w, b, losses = self.experiment["train"](np.array([1.0]), np.array([3.0]), lr=0.1, steps=1)
        self.assertAlmostEqual(w, 0.6)
        self.assertAlmostEqual(b, 0.6)
        np.testing.assert_allclose(losses, [9, 3.24])

    def test_gradients_match_numerical_changes_in_loss(self):
        predict = self.experiment["predict"]
        mse = self.experiment["mse"]
        gradients = self.experiment["gradients"]
        w, b, eps = 0.7, -0.2, 1e-6
        dw, db = gradients(self.x, self.y, w, b)
        numerical_dw = (mse(predict(self.x, w + eps, b), self.y) - mse(predict(self.x, w - eps, b), self.y)) / (2 * eps)
        numerical_db = (mse(predict(self.x, w, b + eps), self.y) - mse(predict(self.x, w, b - eps), self.y)) / (2 * eps)
        self.assertAlmostEqual(dw, numerical_dw, places=6)
        self.assertAlmostEqual(db, numerical_db, places=6)

    def test_training_learns_rule_and_predicts_unseen_inputs(self):
        w, b, losses = self.experiment["train"](self.x, self.y, lr=0.1, steps=400)
        self.assertAlmostEqual(w, 2, places=3)
        self.assertAlmostEqual(b, 1, places=3)
        self.assertLess(losses[-1], losses[0] / 1000)
        unseen = np.array([0.75, 1.75])
        np.testing.assert_allclose(self.experiment["predict"](unseen, w, b), [2.5, 4.5], atol=0.001)

    def test_large_learning_rate_increases_loss(self):
        _, _, losses = self.experiment["train"](self.x, self.y, lr=1.0, steps=20)
        self.assertTrue(np.isfinite(losses).all())
        self.assertGreater(losses[-1], losses[0])

    def test_more_steps_help_with_small_learning_rate(self):
        train = self.experiment["train"]
        _, _, short = train(self.x, self.y, lr=0.001, steps=20)
        _, _, long = train(self.x, self.y, lr=0.001, steps=400)
        self.assertLess(long[-1], short[-1])


if __name__ == "__main__":
    unittest.main()
