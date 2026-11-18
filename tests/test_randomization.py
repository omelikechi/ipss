# Test IPSS randomization
"""Fast tests for per-fit selector seeds and reproducibility via random_state."""
import unittest
from unittest import mock

import numpy as np

from ipss import ipss
from ipss.main import selection

print()


def recording_selector(seen):
	def selector(x, y, random_state=None):
		seen.append((random_state, x[:, 0].copy()))
		return np.arange(x.shape[1], dtype=float)
	return selector


class RandomizationTests(unittest.TestCase):

	def setUp(self):
		self.x = np.arange(60, dtype=float).reshape(20, 3)
		self.y = np.arange(20, dtype=float)

	def test_tree_selector_fits_get_distinct_seeds_by_default(self):
		seen = []
		selector = recording_selector(seen)
		with mock.patch('ipss.main.seeded_selectors', (selector,)):
			np.random.seed(0)
			selection(self.x, self.y, None, 10, selector)
		seeds = [seed for seed, _ in seen]
		self.assertNotIn(None, seeds)
		self.assertNotEqual(seeds[0], seeds[1])

	def test_given_random_state_is_kept(self):
		seen = []
		selector = recording_selector(seen)
		with mock.patch('ipss.main.seeded_selectors', (selector,)):
			selection(self.x, self.y, None, 10, selector, 101, random_state=5)
		self.assertEqual([seed for seed, _ in seen], [5, 5])

	def test_other_selectors_get_no_seed(self):
		seen = []
		selection(self.x, self.y, None, 10, recording_selector(seen), 101)
		self.assertEqual([seed for seed, _ in seen], [None, None])

	def test_subsamples_are_reproducible(self):
		first, second = [], []
		selection(self.x, self.y, None, 10, recording_selector(first), 101)
		selection(self.x, self.y, None, 10, recording_selector(second), 101)
		for (_, left), (_, right) in zip(first, second):
			np.testing.assert_array_equal(left, right)


class IpssReproducibilityTests(unittest.TestCase):

	def setUp(self):
		rng = np.random.default_rng(0)
		self.X = rng.normal(size=(100, 30))
		self.y = self.X[:, 0] + rng.normal(size=100)

	def run_ipss(self, selector, **kwargs):
		return ipss(self.X, self.y, selector=selector, B=10, n_jobs=2, preselector='gb', **kwargs)['efp_scores']

	def test_random_state_reproduces_results(self):
		for selector in ['gb', 'rf', 'ufi']:
			with self.subTest(selector=selector):
				self.assertEqual(self.run_ipss(selector, random_state=7), self.run_ipss(selector, random_state=7))

	def test_global_seed_reproduces_results(self):
		np.random.seed(7)
		first = self.run_ipss('gb')
		np.random.seed(7)
		self.assertEqual(first, self.run_ipss('gb'))

	def test_selector_random_state_is_ignored_with_warning(self):
		with self.assertWarns(UserWarning):
			result = self.run_ipss('gb', random_state=7, selector_args={'max_depth': 1, 'colsample_bynode': 1/3,
				'n_estimators': 100, 'importance_type': 'gain', 'random_state': 0})
		self.assertEqual(result, self.run_ipss('gb', random_state=7))


if __name__ == '__main__':
	unittest.main()
