# Test IPSS correctness
'''Fast regression tests for the E(FP) integrals, the cutoff, plateaus, selector defaults, and edge cases.'''
import unittest
import warnings

import numpy as np
from scipy.sparse import csr_matrix

from ipss import ipss
from ipss.helpers import (categorical_columns, compute_delta, compute_qvalues, correlation_summary, efp_bound_integrand,
	integrate, ipss_function_order, jitter_columns, selector_and_args, stratified_halves)
from ipss.preselection import preselection
from ipss.processing import postprocess_ipss, preprocess_ipss

print()


def fixed_selector(importances):
	importances = np.asarray(importances, dtype=float)
	def selector(X, y):
		return importances.copy()
	return selector


def noisy_selector(X, y):
	# importance = |correlation with y| plus a little noise, so that paths are nontrivial
	corr = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
	return corr + np.random.uniform(0, 0.05, X.shape[1])


def brute_force_qvalues(efp_scores):
	values = list(efp_scores.values())
	fdrs = [(t, max(s for s in values if s <= t) / sum(s <= t for s in values)) for t in values]
	return {j: min(fdr for t, fdr in fdrs if score <= t) for j, score in efp_scores.items()}


class IntegrationTests(unittest.TestCase):

	def test_probability_measure_has_mass_one(self):
		alphas = np.logspace(1, -3, 25)
		for delta in (0, 0.5, 1, 1 - 1e-9, 1.25, 3):
			value, k = integrate(np.ones(25), alphas, delta)
			self.assertAlmostEqual(value, 1.0, places=12)
			self.assertEqual(k, 25)

	def test_riemann_sum_converges_to_exact_integral(self):
		# integral of lambda with respect to mu_delta on [a, b], delta = 3: (b^-1 - a^-1)/(-1) / ((b^-2 - a^-2)/(-2))
		a, b, delta = 0.1, 1.0, 3
		exact = (1 / a - 1 / b) / ((a**-2 - b**-2) / 2)
		alphas = np.logspace(0, -1, 20001)
		value, _ = integrate(alphas, alphas, delta)
		self.assertAlmostEqual(value, exact, places=4)

	def test_cutoff_interval_is_normalized_on_itself(self):
		alphas = np.logspace(0, -8, 100)
		path = np.linspace(0, 1, 100)**2
		for delta in (1, 1.25, 3):
			value, k = integrate(path, alphas, delta, cutoff=0.05)
			self.assertLessEqual(value, 0.05)
			self.assertLess(k, 100)
			same, _ = integrate(path[:k], alphas[:k], delta)
			self.assertAlmostEqual(value, same, places=12)
			longer, _ = integrate(path[:k + 1], alphas[:k + 1], delta)
			self.assertGreater(longer, 0.05)

	def test_two_dimensional_values(self):
		alphas = np.logspace(0, -2, 10)
		values = np.random.default_rng(0).uniform(size=(10, 4))
		joint, _ = integrate(values, alphas, 1.25)
		separate = [integrate(values[:, j], alphas, 1.25)[0] for j in range(4)]
		np.testing.assert_allclose(joint, separate)


class EfpTests(unittest.TestCase):

	def setUp(self):
		rng = np.random.default_rng(1)
		self.X = rng.normal(size=(100, 20))
		self.y = self.X[:, 0] + 0.5 * self.X[:, 1] + rng.normal(size=100)

	def test_efp_uses_same_interval_and_measure_as_bound(self):
		for delta in (1, 1.25, 3):
			out = ipss(self.X, self.y, selector=noisy_selector, preselect=False, B=10, delta=delta,
				random_state=0, _return_details=True)
			alphas, paths, integral = out['alphas'], out['stability_paths'], out['integral']
			self.assertEqual(paths.shape[0], len(alphas))
			self.assertEqual(out['indicators'].shape[0], len(alphas))
			self.assertEqual(len(out['average_selected']), len(alphas))
			self.assertLessEqual(integral, 0.05)
			h = np.where(paths > 0.5, (2 * paths - 1)**3, 0.0)
			scores, _ = integrate(h, alphas, delta)
			for j in range(20):
				expected = min(integral / scores[j], 20) if scores[j] > 0 else 20
				self.assertAlmostEqual(out['efp_scores'][j], expected, places=6)

	def test_temporary_plateau_does_not_end_the_path(self):
		importances = np.zeros(20)
		importances[:2] = [1, 0.01]
		out = ipss(self.X, self.y, selector=fixed_selector(importances), preselect=False, B=5, delta=1.25)
		self.assertEqual(out['stability_paths'][:, 1].max(), 1.0)
		self.assertLess(out['efp_scores'][1], 1)
		self.assertLess(out['efp_scores'][0], 1)
		self.assertTrue(all(out['efp_scores'][j] == 20 for j in range(2, 20)))

	def test_all_zero_importances_give_null_result(self):
		out = ipss(self.X, self.y, selector=fixed_selector(np.zeros(20)), preselect=False, B=5)
		self.assertTrue(all(v == 20 for v in out['efp_scores'].values()))
		self.assertTrue(all(np.isfinite(v) for v in out['q_values'].values()))

	def test_reinsertion_keeps_retained_scores(self):
		# dcor preselection uses no randomness, so the reduced run matches a run on the retained columns alone
		reduced = ipss(self.X, self.y, selector=noisy_selector, preselector='dcor', preselector_args={'n_keep': 5},
			B=10, random_state=0, _return_details=True)
		kept = reduced['preselect_indices']
		self.assertEqual(len(kept), 5)
		direct = ipss(self.X[:, kept], self.y, selector=noisy_selector, preselect=False, B=10, random_state=0)
		self.assertEqual(len(reduced['efp_scores']), 20)
		for j in set(range(20)) - set(kept.tolist()):
			self.assertEqual(reduced['efp_scores'][j], 20)
		for k, j in enumerate(kept):
			expected = direct['efp_scores'][k] if direct['efp_scores'][k] < 5 else 20
			self.assertEqual(reduced['efp_scores'][int(j)], expected)
		self.assertTrue(any(v < 5 for v in reduced['efp_scores'].values()))


class ArgumentTests(unittest.TestCase):

	def test_user_args_merge_into_defaults(self):
		_, args = selector_and_args('logistic_regression', {'max_iter': 1000})
		self.assertEqual((args['penalty'], args['solver'], args['max_iter']), ('l1', 'liblinear', 1000))
		_, args = selector_and_args('gb_regressor', {'n_estimators': 50})
		self.assertEqual((args['max_depth'], args['colsample_bynode'], args['n_estimators']), (1, 1/3, 50))
		_, args = selector_and_args('rf_classifier', {'max_depth': 3})
		self.assertEqual((args['max_features'], args['max_depth']), (1/10, 3))
		_, args = selector_and_args('gb_regressor', {'max_depth': 2})
		self.assertEqual(args['max_depth'], 2)

	def test_l1_classifier_stays_sparse_with_custom_args(self):
		rng = np.random.default_rng(2)
		X = rng.normal(size=(120, 30))
		y = (X[:, 0] + rng.normal(size=120) > 0).astype(int)
		out = ipss(X, y, selector='l1', selector_args={'max_iter': 1000}, B=10, random_state=0, _return_details=True)
		paths = out['stability_paths']
		self.assertTrue(np.any(paths < 1))  # with l2 every coefficient is nonzero, so every path would be 1
		self.assertLess(out['efp_scores'][0], 1)

	def test_constant_column_linear_preselection(self):
		rng = np.random.default_rng(3)
		X = rng.normal(size=(40, 4))
		X[:, 2] = 1.0
		y = X[:, 0] + rng.normal(size=40)
		X_reduced, indices = preselection(X, y, 'lasso', random_state=0)
		self.assertEqual(X_reduced.shape[0], 40)
		self.assertTrue(len(indices) > 0)


class QvalueTests(unittest.TestCase):

	def test_matches_definition_with_ties(self):
		rng = np.random.default_rng(4)
		for _ in range(50):
			scores = {j: float(v) for j, v in enumerate(rng.choice([0.1, 0.5, 1, 2, 5, 20], 30))}
			fast, slow = compute_qvalues(scores), brute_force_qvalues(scores)
			for j in scores:
				self.assertAlmostEqual(fast[j], slow[j], places=12)

	def test_rejects_nonfinite(self):
		with self.assertRaises(ValueError):
			compute_qvalues({0: 1.0, 1: float('nan')})


class ScoreGridTests(unittest.TestCase):

	def setUp(self):
		rng = np.random.default_rng(5)
		self.X = rng.normal(size=(100, 20))
		self.y = self.X[:, 0] + 0.5 * self.X[:, 1] + rng.normal(size=100)

	def test_invariant_to_score_scale(self):
		base = ipss(self.X, self.y, selector=noisy_selector, preselect=False, B=10, random_state=0, _return_details=True)
		for c in (1e-6, 1e-3, 7.5, 1e4):
			def scaled(X, y, c=c):
				return c * noisy_selector(X, y)
			out = ipss(self.X, self.y, selector=scaled, preselect=False, B=10, random_state=0, _return_details=True)
			self.assertEqual(len(out['alphas']), len(base['alphas']))
			for j in range(20):
				self.assertAlmostEqual(out['efp_scores'][j], base['efp_scores'][j], places=6)

	def test_negative_scores_clipped_without_promoting_zeros(self):
		importances = np.zeros(20)
		importances[:3] = [1.0, 0.5, -0.7]
		out = ipss(self.X, self.y, selector=fixed_selector(importances), preselect=False, B=5)
		self.assertEqual(out['stability_paths'][:, 2].max(), 0)  # negative: never selected
		self.assertEqual(out['stability_paths'][:, 5].max(), 0)  # zero: never selected
		self.assertEqual(out['efp_scores'][2], 20)
		self.assertEqual(out['efp_scores'][5], 20)
		self.assertLess(out['efp_scores'][0], 1)
		self.assertLess(out['efp_scores'][1], 1)

	def test_all_nonpositive_scores_give_null_result(self):
		out = ipss(self.X, self.y, selector=fixed_selector(-np.ones(20)), preselect=False, B=5)
		self.assertTrue(all(v == 20 for v in out['efp_scores'].values()))
		self.assertTrue(all(v == 1 for v in out['q_values'].values()))

	def test_selector_outputs_are_not_mutated(self):
		returned = []
		def recording(X, y):
			scores = noisy_selector(X, y) - 0.02
			returned.append((scores, scores.copy()))
			return scores
		ipss(self.X, self.y, selector=recording, preselect=False, B=5)
		self.assertTrue(all(np.array_equal(a, b) for a, b in returned))
		self.assertTrue(any(np.any(a < 0) for a, _ in returned))


class GoldenTests(unittest.TestCase):

	def test_integrate_by_hand(self):
		# grid 8, 4, 2, 1: under delta = 1 each interval has mass 1/3; under delta = 0 the masses are 4/7, 2/7, 1/7
		alphas = np.array([8.0, 4.0, 2.0, 1.0])
		bound = np.array([0.0, 0.03, 0.09, 0.5])
		value, k = integrate(bound, alphas, 1, cutoff=0.05)
		self.assertEqual(k, 4)
		self.assertAlmostEqual(value, 0.04, places=12)  # (0 + 0.03 + 0.09) / 3
		value, k = integrate(bound, alphas, 1, cutoff=0.03)
		self.assertEqual(k, 3)
		self.assertAlmostEqual(value, 0.015, places=12)  # (0 + 0.03) / 2
		value, k = integrate(bound, alphas, 0)
		self.assertAlmostEqual(value, (0 * 4 + 0.03 * 2 + 0.09 * 1) / 7, places=12)

	def test_postprocess_by_hand(self):
		# one complementary pair (B = 1), p = 4, grid 8, 4, 2, 1, h1 (bound integrand q^2 / p)
		alphas = np.array([8.0, 4.0, 2.0, 1.0])
		results = np.zeros((1, 4, 2, 4))
		results[0, 1, :, 0] = 1  # alpha 4: feature 0 in both fits (q = 1)
		results[0, 2, :, :2] = 1  # alpha 2: features 0 and 1 in both fits (q = 2)
		results[0, 3, 0, [0, 1, 2]] = 1  # alpha 1: features 0, 1, 2 in fit 1 and 0, 1, 3 in fit 2 (q = 3)
		results[0, 3, 1, [0, 1, 3]] = 1
		# bound integrand 0, 1/4, 1, 9/4; cutoff 0.5 keeps all three intervals: I = (0 + 1/4 + 1) / 3 = 5/12
		# h1 integrals over upper endpoints: feature 0 (0 + 1 + 1) / 3, feature 1 (0 + 0 + 1) / 3, features 2, 3: 0
		out = postprocess_ipss(results.copy(), alphas, 4, 4, 4, np.arange(4), 1, 0.5, 1, 'h1', None, None)
		expected = {0: (5 / 12) / (2 / 3), 1: (5 / 12) / (1 / 3), 2: 4, 3: 4}
		for j, v in expected.items():
			self.assertAlmostEqual(out['efp_scores'][j], v, places=8)
		for j, v in {0: 0.625, 1: 0.625, 2: 1, 3: 1}.items():
			self.assertAlmostEqual(out['q_values'][j], v, places=8)
		# cutoff 0.2 keeps two intervals: I = (0 + 1/4) / 2; feature 0 integral (0 + 1) / 2; feature 1 has none
		out = postprocess_ipss(results.copy(), alphas, 4, 4, 4, np.arange(4), 1, 0.2, 1, 'h1', None, None)
		for j, v in {0: 0.25, 1: 4, 2: 4, 3: 4}.items():
			self.assertAlmostEqual(out['efp_scores'][j], v, places=8)
		self.assertEqual(out['stability_paths'].shape, (3, 4))


class PreselectionTests(unittest.TestCase):

	def setUp(self):
		rng = np.random.default_rng(6)
		self.X = rng.normal(size=(120, 150))
		self.y = self.X[:, 0] + rng.normal(size=120)

	def test_gb_preselection_honors_n_keep(self):
		_, kept = preselection(self.X, self.y, 'gb_regressor', {'n_keep': 10}, random_state=0)
		self.assertEqual(len(kept), 10)
		_, kept = preselection(self.X, self.y, 'gb_regressor', None, random_state=0)
		self.assertGreaterEqual(len(kept), 100)

	def test_ufi_preselector_is_rejected(self):
		with self.assertRaises(ValueError):
			ipss(self.X, self.y, selector='gb', preselector='ufi', B=2)

	def test_preselect_count_and_fraction(self):
		for preselect, expected in ((7, 7), (0.1, 15), (1.0, 150), (1, 1), (500, 150)):
			out = ipss(self.X, self.y, selector='gb', preselect=preselect, B=2, random_state=0, _return_details=True)
			self.assertEqual(len(out['preselect_indices']), expected, msg=str(preselect))
		out = ipss(self.X, self.y, selector='gb', preselect=True, B=2, random_state=0, _return_details=True)
		self.assertGreaterEqual(len(out['preselect_indices']), 100)  # gb adaptive rule, not one feature
		for bad in (0, -2, 1.5, 0.0, float('nan'), 'all'):
			with self.assertRaises(ValueError, msg=str(bad)):
				ipss(self.X, self.y, selector='gb', preselect=bad, B=2)
		with self.assertRaises(ValueError):
			ipss(self.X, self.y, selector='gb', preselect=10, preselector_args={'n_keep': 5}, B=2)

	def test_rare_class_logistic_preselection(self):
		rng = np.random.default_rng(7)
		X = rng.normal(size=(60, 30))
		y = np.zeros(60)
		y[:3] = 1
		_, kept = preselection(X, y, 'logistic_regression', {'n_runs': 20}, random_state=0)
		self.assertGreater(len(kept), 0)


class ValidationTests(unittest.TestCase):

	def setUp(self):
		rng = np.random.default_rng(8)
		self.X = rng.normal(size=(50, 10))
		self.y = self.X[:, 0] + rng.normal(size=50)

	def test_invalid_arguments_raise(self):
		bad = [
			dict(B=0), dict(B=2.5), dict(n_alphas=1), dict(cutoff=0), dict(cutoff=float('nan')),
			dict(subsample_size=0), dict(subsample_size=-3), dict(subsample_size=0.4), dict(target_fdr=2),
			dict(target_fdr=0), dict(target_fp=-1), dict(target_fp=1, target_fdr=0.1), dict(delta=float('inf')),
			dict(ipss_function='h0'), dict(preselector_args={'n_keep': 0}), dict(preselector_args={'n_runs': 0}),
		]
		for kwargs in bad:
			with self.assertRaises((ValueError, TypeError), msg=str(kwargs)):
				ipss(self.X, self.y, selector='gb', **{'B': 2, **kwargs})

	def test_invalid_data_raise(self):
		X_nan = self.X.copy()
		X_nan[0, 0] = np.nan
		for X, y in ((self.X, self.y[:-1]), (X_nan, self.y), (self.X[:, 0], self.y), (self.X, np.full(50, np.inf))):
			with self.assertRaises(ValueError):
				ipss(X, y, selector='gb', B=2)

	def test_bad_selector_output_raises(self):
		for output in (lambda X, y: np.ones(3), lambda X, y: np.full(X.shape[1], np.nan)):
			with self.assertRaises(ValueError):
				ipss(self.X, self.y, selector=output, preselect=False, B=2)

	def test_degenerate_l1_gives_null_result(self):
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			out = ipss(np.ones((50, 10)), self.y, selector='l1', B=2)
		self.assertTrue(all(v == 10 for v in out['efp_scores'].values()))

	def test_rare_class_error_and_stratify(self):
		rng = np.random.default_rng(9)
		X = rng.normal(size=(40, 10))
		y = np.zeros(40)
		y[:3] = 1
		with self.assertRaisesRegex(ValueError, 'stratify=True'):
			ipss(X, y, selector='gb', preselect=False, B=20, random_state=0)
		out = ipss(X, y, selector='gb', preselect=False, B=20, random_state=0, stratify=True)
		self.assertEqual(len(out['efp_scores']), 10)

	def test_default_sampling_unchanged_by_stratify_flag(self):
		a = ipss(self.X, self.y, selector='gb', B=5, random_state=0)
		b = ipss(self.X, self.y, selector='gb', B=5, random_state=0, stratify=True)  # not binary: no effect
		self.assertEqual(a['efp_scores'], b['efp_scores'])


class JitterTests(unittest.TestCase):
	def setUp(self):
		rng = np.random.default_rng(11)
		self.X = np.hstack([rng.normal(size=(60, 3)), rng.integers(0, 2, (60, 2)), rng.integers(0, 4, (60, 1)) * 0.5,
			np.ones((60, 1))])
		self.y = self.X[:, 0] + self.X[:, 3] + rng.normal(size=60)

	def prep(self, selector, jitter=None):
		return preprocess_ipss(self.X, self.y, selector, None, False, None, None, None, None, None, None, None, None,
			random_state=0, jitter=jitter)

	def test_detects_categorical_columns(self):
		# constant and continuous columns are left alone
		self.assertEqual(categorical_columns(self.X).tolist(), [3, 4, 5])

	def test_categories_stay_separated(self):
		Xj = jitter_columns(self.X, [3, 4, 5], np.random.RandomState(0))
		for j in (3, 4, 5):
			gap = np.min(np.diff(np.unique(self.X[:, j])))
			self.assertLessEqual(np.max(np.abs(Xj[:, j] - self.X[:, j])), 1e-4 * gap)
			self.assertEqual(np.unique(Xj[:, j]).size, 60)
			np.testing.assert_array_equal(np.round(Xj[:, j] / gap) * gap, self.X[:, j])
		np.testing.assert_array_equal(Xj[:, [0, 1, 2, 6]], self.X[:, [0, 1, 2, 6]])

	def test_defaults_by_selector(self):
		self.assertEqual(self.prep('gb')['jitter_indices'].tolist(), [3, 4, 5])
		self.assertEqual(self.prep('rf')['jitter_indices'].tolist(), [3, 4, 5])
		self.assertEqual(self.prep('l1')['jitter_indices'].tolist(), [])
		self.assertEqual(self.prep('l1', jitter=True)['jitter_indices'].tolist(), [3, 4, 5])
		self.assertEqual(self.prep('gb', jitter=False)['jitter_indices'].tolist(), [])
		self.assertEqual(self.prep('gb', jitter=[4, 0])['jitter_indices'].tolist(), [0, 4])

	def test_input_not_mutated_and_reproducible(self):
		X0 = self.X.copy()
		a = ipss(self.X, self.y, selector='gb', B=10, random_state=3)
		b = ipss(self.X, self.y, selector='gb', B=10, random_state=3)
		np.testing.assert_array_equal(self.X, X0)
		self.assertEqual(a['efp_scores'], b['efp_scores'])

	def test_bad_jitter(self):
		for bad in ([1.5], [7], [-1], 'yes', [[0]]):
			with self.assertRaises(ValueError):
				self.prep('gb', jitter=bad)


class StratifiedHalvesTests(unittest.TestCase):
	def check(self, labels, size, seed=0):
		halves = stratified_halves(labels, size, np.random.RandomState(seed))
		self.assertEqual([len(h) for h in halves], [size, size])
		self.assertEqual(len(np.intersect1d(halves[0], halves[1])), 0)
		return halves

	def test_rare_class_exact_size(self):
		labels = np.r_[np.ones(3), np.zeros(37)]
		for half in self.check(labels, 20):
			self.assertEqual(set(labels[half]), {0, 1})

	def test_random_configurations(self):
		rng = np.random.default_rng(0)
		for _ in range(300):
			n = int(rng.integers(4, 80))
			labels = rng.integers(0, int(rng.integers(2, 4)), n)
			size = int(rng.integers(1, n // 2 + 1))
			halves = self.check(labels, size, int(rng.integers(1e6)))
			for c, count in zip(*np.unique(labels, return_counts=True)):
				share = [np.sum(labels[h] == c) for h in halves]
				self.assertLessEqual(abs(share[0] - share[1]), 1)
				if count >= 2 and size >= len(np.unique(labels)):
					self.assertTrue(all(k >= 1 for k in share))

	def test_bad_size(self):
		with self.assertRaises(ValueError):
			stratified_halves(np.r_[np.ones(5), np.zeros(5)], 6, np.random.RandomState(0))


class DeltaAndSelectorTests(unittest.TestCase):
	def setUp(self):
		rng = np.random.default_rng(12)
		self.X = rng.normal(size=(60, 8))
		self.y = self.X[:, 0] + rng.normal(size=60)

	def prep(self, **kwargs):
		args = dict(selector='gb', selector_args=None, preselect=False, preselector=None, preselector_args=None, B=None,
			n_alphas=None, ipss_function=None, delta=None, standardize_X=None, center_y=None)
		args.update(kwargs)
		return preprocess_ipss(self.X, self.y, **args, random_state=0)

	def test_default_delta(self):
		# 1 for importance selectors; set from the feature correlations for l1
		for selector in ('gb', 'rf', 'adaptive_lasso'):
			self.assertEqual(self.prep(selector=selector)['delta'], 1)
		self.assertEqual(self.prep(selector='l1')['delta'], compute_delta(self.X, 'lasso'))
		self.assertEqual(self.prep(selector='l1', delta=1.5)['delta'], 1.5)

	def test_compute_delta_matches_ipss_1_1_10(self):
		# reference: the 1.1.10 rule, written with np.corrcoef and the linear system it solved
		def reference(X):
			cor = np.abs(np.corrcoef(X, rowvar=False))
			np.fill_diagonal(cor, 0)
			avg_cor, avg_max = np.mean(cor), np.mean(np.max(cor, axis=1))
			if avg_cor <= 1 / 20:
				return 1
			if avg_cor >= 1 / 4:
				return 0
			coefs = np.linalg.inv(np.array([[1, 1 / 20, 1 / 2], [1, 1 / 4, 1 / 2], [1, 1 / 20, 1]])) @ np.array([1, 0, 0])
			return max(0, min(1, coefs @ np.array([1, avg_cor, avg_max])))
		rng = np.random.default_rng(0)
		idx = np.arange(40)
		for rho in (0, 0.5, 0.7, 0.8, 0.95):
			chol = np.linalg.cholesky(rho ** np.abs(idx[:, None] - idx[None, :]))
			X = rng.normal(size=(300, 40)) @ chol.T
			for selector in ('lasso', 'logistic_regression'):
				self.assertAlmostEqual(compute_delta(X, selector), reference(X), places=10)
			self.assertEqual(compute_delta(X, 'rf_regressor'), 1)
		# strong correlation lowers delta; independent features with large n keep delta = 1
		X = rng.normal(size=(2000, 40))
		self.assertEqual(compute_delta(X, 'lasso'), 1)
		X = rng.normal(size=(300, 40)) @ np.linalg.cholesky(0.8 ** np.abs(idx[:, None] - idx[None, :])).T
		self.assertLess(compute_delta(X, 'lasso'), 0.5)

	def test_correlation_summary_blocks_sparse_and_constants(self):
		rng = np.random.default_rng(1)
		X = rng.normal(size=(50, 13)) * (rng.random((50, 13)) < 0.3)
		X[:, 4] = 2.0
		full = correlation_summary(X)
		self.assertTrue(np.all(np.isfinite(full)))
		for block_size in (1, 3, 13):
			np.testing.assert_allclose(correlation_summary(X, block_size), full, rtol=1e-12)
		np.testing.assert_allclose(correlation_summary(csr_matrix(X)), full, rtol=1e-9)
		X_copy = X.copy()
		correlation_summary(X)
		np.testing.assert_array_equal(X, X_copy)
		self.assertEqual(compute_delta(np.ones((10, 3)), 'lasso'), 1)

	def test_delta_forms(self):
		self.assertEqual(self.prep(delta=np.float64(2))['delta'], 2)
		self.assertEqual(self.prep(delta=np.array(2.0))['delta'], 2)
		self.assertEqual(self.prep(delta=[1, np.int64(2)])['delta'], [1, 2])
		out = ipss(self.X, self.y, selector='gb', B=5, delta=np.array(1.5), random_state=0)
		self.assertEqual(len(out['efp_scores']), 8)
		for bad in ([1, 2], 'a', np.nan):
			with self.assertRaises(ValueError):
				ipss(self.X, self.y, selector='gb', B=5, delta=bad)

	def test_unknown_selector_and_one_feature(self):
		with self.assertRaisesRegex(ValueError, 'Unrecognized selector'):
			ipss(self.X, self.y, selector='bogus')
		with self.assertRaisesRegex(ValueError, '2 features'):
			ipss(self.X[:, :1], self.y, selector='l1')


class IpssFunctionOrderTests(unittest.TestCase):
	def setUp(self):
		rng = np.random.default_rng(13)
		self.X = rng.normal(size=(60, 8))
		self.y = self.X[:, 0] + rng.normal(size=60)

	def test_general_bound_matches_paper_bounds(self):
		p, B = 300, 100
		q = np.linspace(0.5, 60, 40)
		paper = {1: q**2 / p, 2: q**2 / (p * B) + (B - 1) * q**4 / (B * p**3),
			3: q**2 / (p * B**2) + 3 * (B - 1) * q**4 / (p**3 * B**2) + (B - 1) * (B - 2) * q**6 / (p**5 * B**2)}
		for m, bound in paper.items():
			np.testing.assert_allclose(efp_bound_integrand(q, p, B, m), bound, rtol=1e-12)
		# m = 4: S(4, k) = 1, 7, 6, 1
		r = q / p
		bound4 = p * (r**2 * B + 7 * B * (B - 1) * r**4 + 6 * B * (B - 1) * (B - 2) * r**6
			+ B * (B - 1) * (B - 2) * (B - 3) * r**8) / B**4
		np.testing.assert_allclose(efp_bound_integrand(q, p, B, 4), bound4, rtol=1e-12)

	def test_order_parsing(self):
		self.assertEqual(ipss_function_order('h1'), 1)
		self.assertEqual(ipss_function_order('h12'), 12)
		for bad in ('h0', 'h', 'H3', 'h3.5', 3, None):
			with self.assertRaises(ValueError):
				ipss_function_order(bad)
		with self.assertRaisesRegex(ValueError, 'ipss_function'):
			ipss(self.X, self.y, selector='gb', B=5, ipss_function='h0')

	def test_high_orders_run(self):
		out = ipss(self.X, self.y, selector='gb', B=5, ipss_function='h4', random_state=0)
		self.assertEqual(len(out['efp_scores']), 8)

	def test_defaults_unchanged(self):
		for selector, expected in (('l1', 'h2'), ('adaptive_lasso', 'h3'), ('gb', 'h3'), ('rf', 'h3')):
			prep = preprocess_ipss(self.X, self.y, selector, None, False, None, None, None, None, None, None, None, None,
				random_state=0)
			self.assertEqual(prep['ipss_function'], expected)

	def test_default_preselectors(self):
		from ipss.helpers import resolve_preselector
		for selector, expected in (('gb_regressor', 'gb_regressor'), ('gb_classifier', 'gb_classifier'), ('lasso', 'lasso'),
				('adaptive_lasso_regressor', 'adaptive_lasso_regressor'), ('rf_regressor', 'dcor'), ('ufi_regressor', 'dcor')):
			self.assertEqual(resolve_preselector(None, selector, False), expected)


class ReviewFixTests(unittest.TestCase):
	def setUp(self):
		rng = np.random.default_rng(0)
		self.X = rng.normal(size=(80, 8))
		self.y = self.X[:, 0] + rng.normal(size=80)

	def test_internal_selector_names_get_public_defaults(self):
		for internal, public in (('lasso', 'l1'), ('adaptive_lasso_regressor', 'adaptive_lasso'), ('gb_regressor', 'gb')):
			a = ipss(self.X, self.y, selector=internal, B=5, random_state=0)
			b = ipss(self.X, self.y, selector=public, B=5, random_state=0)
			self.assertEqual(a['efp_scores'], b['efp_scores'])

	def test_custom_preselector_is_reproducible(self):
		def random_preselector(X, y):
			return np.random.rand(X.shape[1])
		runs = [ipss(self.X, self.y, selector='gb', B=5, preselector=random_preselector, preselect=4, random_state=7)
			for _ in range(2)]
		self.assertEqual(runs[0]['efp_scores'], runs[1]['efp_scores'])

	def test_order_limit(self):
		self.assertEqual(ipss_function_order('h100'), 100)
		with self.assertRaisesRegex(ValueError, 'h100'):
			ipss_function_order('h101')

	def test_null_result_paths_are_an_array(self):
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			out = ipss(self.X, self.y, selector=fixed_selector(np.zeros(8)), B=5, random_state=0)
		self.assertIsInstance(out['stability_paths'], np.ndarray)

	def test_uncorrelated_features_give_null_result(self):
		# orthogonal +-1 columns: X'y = 0 exactly for a continuous y
		h = np.array([[1, 1, 1, 1, -1, -1, -1, -1], [1, 1, -1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, -1],
			[1, -1, -1, 1, 1, -1, -1, 1]], dtype=float)
		X, y = np.tile(h[:2].T, (3, 1)), np.tile(h[2] + 0.5 * h[3], 3)
		with self.assertWarnsRegex(UserWarning, 'correlated'):
			out = ipss(X, y, selector='l1', B=5, standardize_X=False, random_state=0)
		self.assertEqual(out['selected_features'], [])

	def test_argument_types(self):
		for kwargs in ({'selector_args': 3}, {'preselector_args': 'a'}, {'standardize_X': 'no'}, {'stratify': 'yes'}):
			with self.assertRaises(ValueError):
				ipss(self.X, self.y, selector='l1', B=5, random_state=0, **kwargs)

	def test_ufi_requires_default_bootstrap(self):
		for args in ({'bootstrap': False}, {'max_samples': 0.5}):
			with self.assertRaisesRegex(ValueError, 'bootstrap'):
				ipss(self.X, self.y, selector='ufi', selector_args=args, B=2, random_state=0)


if __name__ == '__main__':
	unittest.main()
