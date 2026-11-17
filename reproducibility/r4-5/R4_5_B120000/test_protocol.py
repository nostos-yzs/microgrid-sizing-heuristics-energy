"""Small contract tests; no original algorithm training is executed here."""
import math
import time
import unittest
import numpy as np
from worker import BudgetEnv, BudgetExhausted
from analysis import mean_ci, wilson, holm, wilcoxon_p

class FakeEMS:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0
    def evaluate_solution(self, x):
        self.calls += 1
        value, status = next(self.outputs)
        return value, {'status': status}

class ProtocolTests(unittest.TestCase):
    def test_hard_budget_survives_legacy_exception_handler(self):
        original = FakeEMS([(3., 'success'), (2., 'success')])
        env = BudgetEnv(original, 2, 2.5, 1, time.perf_counter())
        env.evaluate_solution([1]); env.evaluate_solution([2])
        with self.assertRaises(BudgetExhausted):
            try:
                env.evaluate_solution([3])
            except Exception:
                self.fail('Legacy handler must not swallow the budget stop.')
        self.assertEqual(original.calls, 2)
        self.assertEqual(env.calls, 2)
        self.assertEqual(env.best, 2.)
        self.assertEqual(env.best_x.tolist(), [2.])
        self.assertEqual(env.target_call, 2)

    def test_penalty_or_invalid_status_is_not_a_feasible_incumbent(self):
        outputs = [(1., 'infeasible'), (float('nan'), 'success'),
                   (1e6, 'success'), (4., 'success')]
        env = BudgetEnv(FakeEMS(outputs), 4, 3., 1, time.perf_counter())
        for i in range(4): env.evaluate_solution([i])
        self.assertEqual(env.best, 4.)
        self.assertEqual(env.best_x.tolist(), [3.])
        self.assertIsNone(env.target_call)
        self.assertIsNone(env.target_seconds)
        self.assertEqual(len(env.improvements), 1)

    def test_fully_infeasible_run_retains_missing_endpoint(self):
        env = BudgetEnv(FakeEMS([(1., 'failed')]), 1, .5, 1, time.perf_counter())
        env.evaluate_solution([0])
        self.assertIsNone(env.best_x)
        self.assertTrue(math.isinf(env.best))
        self.assertIsNone(env.target_call)

    def test_mean_confidence_interval_hand_calculation(self):
        n, m, sd, low, high = mean_ci([1., 2., 3.])
        self.assertEqual(n, 3)
        self.assertEqual(m, 2.)
        self.assertEqual(sd, 1.)
        half = 4.302652729911275 / math.sqrt(3)
        self.assertAlmostEqual(low, 2-half, places=9)
        self.assertAlmostEqual(high, 2+half, places=9)
        self.assertTrue(math.isnan(mean_ci([4])[3]))

    def test_holm_known_family_and_missing_tests(self):
        np.testing.assert_allclose(holm([.01, .04, .03, np.nan]), [.03, .06, .06, np.nan])

    def test_wilson_boundary(self):
        low, high = wilson(0, 22)
        self.assertAlmostEqual(low, 0., places=12)
        self.assertAlmostEqual(high, .1486548746086208, places=10)
        low2, high2 = wilson(22, 22)
        self.assertAlmostEqual(high2, 1., places=12)
        self.assertAlmostEqual(low2, 1-high, places=12)

    def test_identical_pairs_have_no_evidence_of_difference(self):
        self.assertEqual(wilcoxon_p([0]*22), 1.)

if __name__ == '__main__':
    unittest.main(verbosity=2)
