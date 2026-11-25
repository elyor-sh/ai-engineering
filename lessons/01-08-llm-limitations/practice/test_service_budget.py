"""Проверки учебного бюджета и метрик. Реальные сервисы не вызываются."""

import runpy
import unittest
from pathlib import Path


class ServiceBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name("01_service_budget.py")
        assert source.is_file(), "Калькулятор сервиса ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_call_price_matches_lesson(self):
        price = self.experiment["price_per_call"](2000, 400, 0.5, 2.0)
        self.assertAlmostEqual(price, 0.0018)

    def test_zero_output_and_free_tariff(self):
        price = self.experiment["price_per_call"]
        self.assertAlmostEqual(price(2000, 0, 0.5, 2.0), 0.001)
        self.assertEqual(price(2000, 400, 0, 0), 0)

    def test_monthly_baseline_matches_lesson(self):
        self.assertAlmostEqual(self.experiment["monthly_budget"](10_000, 30, 0.0018), 540)

    def test_extra_paid_attempts_are_relative_to_initial_requests(self):
        result = self.experiment["monthly_budget"](10_000, 30, 0.0018, extra_attempts_per_request=0.1)
        self.assertAlmostEqual(result, 594)

    def test_zero_traffic_has_zero_budget(self):
        self.assertEqual(self.experiment["monthly_budget"](0, 30, 0.0018), 0)

    def test_invalid_cost_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.experiment["price_per_call"](-1, 400, 0.5, 2.0)
        with self.assertRaises(ValueError):
            self.experiment["monthly_budget"](100, 30, 0.0018, extra_attempts_per_request=-0.1)
        with self.assertRaises(ValueError):
            self.experiment["price_per_call"](2000, 400, float("nan"), 2.0)

    def test_mean_does_not_hide_slow_tail_in_percentile(self):
        result = self.experiment["latency_summary"]([1.0] * 18 + [15.0] * 2)
        self.assertAlmostEqual(result["mean"], 2.4)
        self.assertEqual(result["median"], 1)
        self.assertEqual(result["p95"], 15)

    def test_nearest_rank_method_is_explicit(self):
        percentile = self.experiment["nearest_rank_percentile"]
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2)
        self.assertEqual(percentile([4, 1, 3, 2], 1.0), 4)

    def test_empty_invalid_or_nonfinite_latencies_are_rejected(self):
        for values in [[], [-1.0, 2.0], [1.0, float("inf")]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.experiment["latency_summary"](values)
        with self.assertRaises(ValueError):
            self.experiment["nearest_rank_percentile"]([1.0], 0)

    def test_conditional_chain_success_matches_lesson(self):
        self.assertAlmostEqual(self.experiment["conditional_chain_success"]([0.95, 0.95, 0.95]), 0.857375)

    def test_chain_success_respects_zero_and_one(self):
        chain = self.experiment["conditional_chain_success"]
        self.assertEqual(chain([1.0, 0.0, 1.0]), 0)
        self.assertEqual(chain([1.0, 1.0, 1.0]), 1)

    def test_invalid_or_empty_conditional_rates_are_rejected(self):
        for rates in [[], [1.1], [-0.1], [float("nan")]]:
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                self.experiment["conditional_chain_success"](rates)


if __name__ == "__main__":
    unittest.main()
