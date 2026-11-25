"""Проверки учебного отбора. Профили вымышленные, API не вызывается."""

import runpy
import unittest
from dataclasses import replace
from pathlib import Path


class ModelSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name("01_model_selection.py")
        assert source.is_file(), "Учебный селектор ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def setUp(self):
        self.candidates = self.experiment["demo_candidates"]()
        self.fast = next(candidate for candidate in self.candidates if candidate.name == "demo-cloud-fast")
        self.requirements = self.experiment["Requirements"](
            min_quality=0.90,
            max_p95_seconds=5.0,
            max_monthly_cost=1000.0,
            input_tokens=10_000,
            output_tokens=1000,
            allow_cloud=True,
            require_structured_output=True,
        )

    def test_strict_shortlist_has_only_fast_demo(self):
        result = self.experiment["shortlist"](self.candidates, self.requirements)
        self.assertEqual([candidate.name for candidate in result], ["demo-cloud-fast"])

    def test_cloud_is_rejected_when_transmission_is_forbidden(self):
        requirements = replace(self.requirements, allow_cloud=False)
        self.assertTrue(self.experiment["rejection_reasons"](self.fast, requirements))

    def test_low_quality_cannot_be_compensated_by_zero_cost(self):
        candidate = replace(self.fast, quality=0.89, monthly_cost=0)
        self.assertTrue(self.experiment["rejection_reasons"](candidate, self.requirements))

    def test_latency_and_cost_are_independent_hard_limits(self):
        candidate = replace(self.fast, p95_seconds=6, monthly_cost=1100)
        self.assertEqual(len(self.experiment["rejection_reasons"](candidate, self.requirements)), 2)

    def test_separate_input_and_output_limits_are_checked(self):
        candidate = replace(self.fast, input_limit=9000, output_limit=900)
        self.assertEqual(len(self.experiment["rejection_reasons"](candidate, self.requirements)), 2)

    def test_combined_context_limit_is_checked_even_when_separate_limits_fit(self):
        candidate = replace(self.fast, context_limit=10_500)
        self.assertTrue(self.experiment["rejection_reasons"](candidate, self.requirements))

    def test_required_structured_output_cannot_be_ignored(self):
        candidate = replace(self.fast, supports_structured_output=False)
        self.assertTrue(self.experiment["rejection_reasons"](candidate, self.requirements))

    def test_exact_boundaries_are_accepted(self):
        candidate = replace(
            self.fast,
            quality=0.90,
            p95_seconds=5.0,
            monthly_cost=1000.0,
            input_limit=10_000,
            output_limit=1000,
            context_limit=11_000,
        )
        self.assertEqual(self.experiment["rejection_reasons"](candidate, self.requirements), [])

    def test_no_candidate_does_not_trigger_cloud_fallback(self):
        requirements = replace(self.requirements, allow_cloud=False)
        self.assertEqual(self.experiment["shortlist"](self.candidates, requirements), [])

    def test_pareto_front_keeps_distinct_tradeoffs(self):
        requirements = replace(
            self.requirements,
            min_quality=0.85,
            max_p95_seconds=10,
            max_monthly_cost=3000,
            input_tokens=6000,
            require_structured_output=False,
        )
        feasible = self.experiment["shortlist"](self.candidates, requirements)
        front = self.experiment["pareto_front"](feasible)
        self.assertEqual([candidate.name for candidate in front], ["demo-local", "demo-cloud-fast", "demo-cloud-large"])

    def test_equal_metrics_do_not_dominate_each_other(self):
        other = replace(self.fast, name="equal-demo")
        self.assertFalse(self.experiment["dominates"](self.fast, other))
        self.assertEqual(len(self.experiment["pareto_front"]([self.fast, other])), 2)

    def test_invalid_metrics_are_rejected_instead_of_silently_passing(self):
        with self.assertRaises(ValueError):
            replace(self.fast, quality=float("nan"))
        with self.assertRaises(ValueError):
            replace(self.requirements, min_quality=1.1)

    def test_empty_inputs_stay_empty(self):
        self.assertEqual(self.experiment["shortlist"]([], self.requirements), [])
        self.assertEqual(self.experiment["pareto_front"]([]), [])


if __name__ == "__main__":
    unittest.main()
