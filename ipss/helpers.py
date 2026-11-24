# Helper functions for IPSS

import re
import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import Lasso, LogisticRegression

from .base_selectors import *

def check_condition(Z, true_features, average_selected, alphas, delta, m, preselect_indices, p_full):
	n_alphas, n_steps, p = Z.shape
	n_blocks = n_steps // (2 * m)
	usable_steps = n_blocks * (2 * m)
	if usable_steps < n_steps:
		Z = Z[:, :usable_steps, :]  # drop incomplete tail

	# compute empirical probabilities
	prob = np.zeros((n_alphas, p))
	for a in range(n_alphas):
		for j in range(p):
			count = 0
			for block_start in range(0, usable_steps, 2 * m):
				block = Z[a, block_start:block_start + 2 * m, j]
				if np.all(block):
					count += 1
			prob[a, j] = count / n_blocks if n_blocks > 0 else 0

	# integrate over alphas for each (retained) feature
	integral_vals = np.array([integrate(prob[:, j], alphas, delta=delta)[0] for j in range(p)])

	# expand back to the full feature space if preselection was used
	all_integrals = np.zeros(p_full)
	all_integrals[preselect_indices] = integral_vals
	integral_vals = all_integrals

	# null features in the full space
	null_features = np.setdiff1d(np.arange(p_full), true_features)

	# empirical left-hand side
	max_null = integral_vals[null_features].max() if len(null_features) > 0 else 0

	# theoretical bound
	upper_bound = integrate((average_selected / p) ** (2 * m), alphas, delta=delta)[0]

	return integral_vals, max_null, upper_bound

def resolve_selector(selector, binary_response):
	if selector == 'adaptive_lasso':
		return 'adaptive_lasso_classifier' if binary_response else 'adaptive_lasso_regressor'
	elif selector == 'l1':
		return 'logistic_regression' if binary_response else 'lasso'
	elif selector == 'rf':
		return 'rf_classifier' if binary_response else 'rf_regressor'
	elif selector == 'gb':
		return 'gb_classifier' if binary_response else 'gb_regressor'
	elif selector == 'ufi':
		return 'ufi_classifier' if binary_response else 'ufi_regressor'
	return selector

def resolve_preselector(preselector, selector, binary_response):
	# default: the selector's own preselection for gb and (adaptive) lasso, and dcor for every other selector.
	# For gb, dcor preselection exceeded FDR targets several-fold on correlated RNA-seq features at large p (it keeps
	# every gene that is correlated with a true one), while gb preselection kept FDR near or below target (it is more
	# conservative there, finding fewer true features)
	if preselector is None:
		own_selectors = ('lasso', 'logistic_regression', 'adaptive_lasso_classifier', 'adaptive_lasso_regressor',
			'gb_classifier', 'gb_regressor')
		if selector in own_selectors:
			return selector
		return 'dcor'
	if not isinstance(preselector, str):
		return preselector
	if preselector in ('ufi', 'ufi_classifier', 'ufi_regressor'):
		raise ValueError("'ufi' is not available as a preselector (it was an ordinary random forest); use 'rf', 'gb', "
			"'dcor', or a custom preselector. selector='ufi' is unaffected.")
	if preselector == 'adaptive_lasso':
		return 'adaptive_lasso_classifier' if binary_response else 'adaptive_lasso_regressor'
	elif preselector == 'l1':
		return 'logistic_regression' if binary_response else 'lasso'
	elif preselector == 'rf':
		return 'rf_classifier' if binary_response else 'rf_regressor'
	elif preselector == 'gb':
		return 'gb_classifier' if binary_response else 'gb_regressor'
	return preselector

def check_response_type(y, selector):
	binary_response = len(np.unique(y)) == 2
	return binary_response, resolve_selector(selector, binary_response)

def compute_alphas(X, y, n_alphas, max_features, binary_response=False):
	n, p = X.shape
	if binary_response:
		y_mean = np.mean(y)
		scaled_residuals = y - y_mean * (1 - y_mean)
		alpha_max = 5 / np.max(np.abs(np.dot(X.T, scaled_residuals) / n))
		selector = LogisticRegression(penalty='l1', solver='liblinear', tol=1e-3, warm_start=True, class_weight='balanced')
		if not np.isfinite(alpha_max):
			alpha_max = 100
		alpha_max = compute_alpha_max_logistic(selector, X, y, alpha_max)
		alpha_min = alpha_max * 1e-10
		test_alphas = np.logspace(np.log10(alpha_max/2), np.log10(alpha_min), 100)
	else:
		alpha_max = 2 * np.max(np.abs(np.dot(X.T,y))) / n
		selector = Lasso(warm_start=True)
		if np.isnan(alpha_max):
			alpha_max = 100
		alpha_min = alpha_max * 1e-10
		test_alphas = np.logspace(np.log10(alpha_max/2), np.log10(alpha_min), 100)
	for alpha in test_alphas:
		if binary_response:
			selector.set_params(C=1/alpha)
		else:
			selector.set_params(alpha=alpha)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			selector.fit(X,y)
		num_selected = np.sum(selector.coef_ != 0)
		if num_selected >= max_features:
			alpha_min = alpha
			break
	alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
	return alphas

# smallest alpha (to within a factor of tol) at which l1 logistic regression selects no features
def compute_alpha_max_logistic(selector, X, y, alpha_start, tol=1.1, max_steps=50):
	def n_selected(alpha):
		selector.set_params(C=1/alpha)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			selector.fit(X, y)
		return np.sum(selector.coef_ != 0)

	# bracket the threshold: features are selected at lower, none at upper
	if n_selected(alpha_start) > 0:
		upper = alpha_start
		for _ in range(max_steps):
			upper *= 2
			if n_selected(upper) == 0:
				break
		else:
			return alpha_start
		lower = upper / 2
	else:
		lower = alpha_start
		for _ in range(max_steps):
			lower /= 2
			if n_selected(lower) > 0:
				break
		else:
			return alpha_start
		upper = lower * 2

	# bisect on a log scale
	while upper / lower > tol:
		mid = np.sqrt(lower * upper)
		if n_selected(mid) > 0:
			lower = mid
		else:
			upper = mid
	return upper

# order m of the ipss function h_m(x) = (2x - 1)^m for x > 0.5 (else 0), given as 'h1', 'h2', ...
def ipss_function_order(ipss_function):
	if isinstance(ipss_function, str) and re.fullmatch(r'h[1-9][0-9]*', ipss_function):
		m = int(ipss_function[1:])
		if m <= 100:
			return m
	raise ValueError(f"ipss_function must be 'h1', 'h2', ..., 'h100' (h followed by an integer from 1 to 100), but got "
		f'{ipss_function!r}.')

# integrand of the E(FP) bound I(Lambda) for h_m: p * sum_k S(m, k) B(B - 1)...(B - k + 1) (q / p)^(2k) / B^m, where
# q is the average number of selected features and S(m, k) are Stirling numbers of the second kind (under the
# assumptions of the IPSS paper); m = 1, 2, 3 give the bounds written out in the paper
def efp_bound_integrand(average_selected, p, B, m):
	r = np.asarray(average_selected, dtype=np.float64) / p
	stirling = [[1]]
	for i in range(1, m + 1):
		stirling.append([0] + [k * (stirling[i - 1][k] if k < i else 0) + stirling[i - 1][k - 1] for k in range(1, i + 1)])
	bound = np.zeros_like(r)
	falling = 1.0
	for k in range(1, m + 1):
		falling *= (B - k + 1) / B
		bound += stirling[m][k] * falling * r ** (2 * k) / B ** (m - k)
	return p * bound

# default delta for every selector except the l1 selectors, whose delta is set from the data (see compute_delta)
default_delta = 1

# delta for the l1 selectors ('lasso' and 'logistic_regression'), set from the feature correlations as in ipss 1.1.10:
# 1 if the average absolute correlation is at most 1/20, 0 if it is at least 1/4, and otherwise
# 2.25 - 5 * average - 2 * average maximum, clipped to [0, 1]. With strongly correlated features, delta = 1 can exceed
# the target FDR (e.g., Toeplitz covariance 0.8^|j-k|); the smaller delta restores control. Every other selector uses
# default_delta
def compute_delta(X, selector):
	if selector not in ('lasso', 'logistic_regression'):
		return default_delta
	avg_cor, avg_max = correlation_summary(X)
	if avg_cor <= 1 / 20:
		return 1
	if avg_cor >= 1 / 4:
		return 0
	return float(np.clip(2.25 - 5 * avg_cor - 2 * avg_max, 0, 1))

# average absolute correlation between features (over all p^2 pairs, with the diagonal counted as zero, as in ipss
# 1.1.10) and average over features of the maximum absolute correlation with another feature. Constant features have
# zero correlation with every feature. Columns are processed in blocks, so memory is O(n p + block_size * p)
def correlation_summary(X, block_size=None):
	n, p = X.shape
	if issparse(X):
		X = X.tocsc()
		mean = np.asarray(X.mean(axis=0)).ravel()
		second_moment = np.asarray(X.multiply(X).mean(axis=0)).ravel()
		std = np.sqrt(np.maximum(second_moment - mean ** 2, 0))
	else:
		X = X - X.mean(axis=0)
		std = X.std(axis=0)
	scale = np.divide(1, std, out=np.zeros(p), where=std > 0)
	block_size = block_size or max(1, 2 ** 22 // p)
	total, max_cor = 0.0, np.zeros(p)
	# errstate silences spurious matmul warnings from Apple's Accelerate BLAS
	with np.errstate(all='ignore'):
		for start in range(0, p, block_size):
			stop = min(start + block_size, p)
			if issparse(X):
				cov = (X[:, start:stop].T @ X).toarray() / n - np.outer(mean[start:stop], mean)
			else:
				cov = X[:, start:stop].T @ X / n
			cor = np.minimum(np.abs(cov * np.outer(scale[start:stop], scale)), 1)
			cor[np.arange(stop - start), np.arange(start, stop)] = 0
			total += cor.sum()
			max_cor[start:stop] = cor.max(axis=1)
	return total / p ** 2, max_cor.mean()

# q-value of each feature: q(j) = min over t >= efp(j) of t / #{k : efp(k) <= t}
def compute_qvalues(efp_scores):
	features = list(efp_scores.keys())
	scores = np.array([efp_scores[feature] for feature in features], dtype=np.float64)
	if scores.size == 0:
		return {}
	if not np.all(np.isfinite(scores)):
		raise ValueError('efp scores must be finite to compute q-values.')
	thresholds = np.unique(scores)
	n_selected = np.searchsorted(np.sort(scores), thresholds, side='right')
	fdrs = np.minimum.accumulate((thresholds / n_selected)[::-1])[::-1]
	q_values = fdrs[np.searchsorted(thresholds, scores)]
	return {feature: float(q) for feature, q in zip(features, q_values)}

# unnormalized mass of mu_delta(dlambda) ~ lambda^(-delta) dlambda on each grid interval [alphas[i], alphas[i-1]],
# rescaled by a common factor for numerical stability
def interval_masses(alphas, delta=1):
	alphas = np.asarray(alphas, dtype=np.float64)
	if alphas.ndim != 1 or alphas.size < 2 or not np.all(alphas > 0):
		raise ValueError('alphas must be a one-dimensional grid of at least two positive values.')
	upper, lower = alphas[:-1], alphas[1:]
	if not np.all(upper > lower):
		raise ValueError('alphas must be strictly decreasing.')
	log_ratio = np.log(upper / lower)
	e = 1 - delta
	if e == 0:
		log_mass = np.log(log_ratio)
	else:
		# integral of lambda^(-delta) over [lower, upper] = lower^e * (exp(e * log_ratio) - 1) / e, in logs
		log_mass = e * np.log(lower) + np.log(np.expm1(e * log_ratio) / e)
	return np.exp(log_mass - np.max(log_mass))

"""
Integral of a path over [alphas[-1], alphas[0]] with respect to the probability measure mu_delta normalized
on that interval. Each grid interval [alphas[i], alphas[i-1]] gets its exact mu_delta mass and the value of
the path at its upper endpoint, alphas[i-1] (the Riemann sum of Proposition S1.1 in the IPSS supplement, with
weights that sum exactly to 1). This is the exact integral with respect to a discrete probability measure on
the grid, so bounds that hold for any probability measure hold exactly for it.

values: path evaluated at each alpha, shape (n_alphas,) or (n_alphas, p)
cutoff: if given, the interval is shortened to [alphas[k-1], alphas[0]] for the smallest alpha (largest k) reached
	before the integral, normalized on the shortened interval, first exceeds the cutoff (Equation 3.3)

Returns the integral and k, the number of grid points in the interval used.
"""
def integrate(values, alphas, delta=1, cutoff=None):
	values = np.asarray(values, dtype=np.float64)
	masses = interval_masses(alphas, delta)
	weighted = np.tensordot(masses, values[:-1], axes=(0, 0))
	if cutoff is None:
		return weighted / np.sum(masses), len(alphas)
	if values.ndim != 1:
		raise ValueError('a cutoff requires a one-dimensional path.')
	averages = np.cumsum(masses * values[:-1]) / np.cumsum(masses)
	exceeded = np.flatnonzero(averages > cutoff)
	n_intervals = max(1, exceeded[0]) if exceeded.size > 0 else len(masses)
	return averages[n_intervals - 1], n_intervals + 1

def return_null_result(p, runtime):
	efp_scores = {j: p for j in range(p)}
	q_values = {j: 1 for j in range(p)}
	return {'efp_scores': efp_scores, 'q_values':q_values, 'runtime':runtime, 'selected_features':[], 'stability_paths':np.zeros((0, p))}

# threshold importance scores on a grid that scales with the scores, so that selections do not depend on their units
"""
results: importance scores, shape (B, 2, p). Negative scores are clipped to 0, so a feature is selected at
threshold alpha only if its score is positive and greater than alpha. The grid runs from just above the largest
score down to 1e-8 times that value; its first point is above every score, so nothing is selected there. If no
score is positive, nothing is ever selected and the result is the null result.
"""
def score_based_selection(results, n_alphas, margin=1.01):
	results = np.clip(np.asarray(results, dtype=np.float64), 0, None)
	max_score = np.max(results) if results.size > 0 else 0.0
	scale = max_score if max_score > 0 else 1.0
	alpha_max = margin * scale
	alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max / 1e8), n_alphas)
	# indicators[b, i, j, :] = 1 if the score of fit j in step b exceeds alphas[i]
	indicators = (results[:, None, :, :] > alphas[None, :, None, None]).astype(np.uint8)
	return indicators, alphas

def selector_and_args(selector, selector_args):
	selectors = {
		'adaptive_lasso_classifier':fit_adaptive_lasso_classifier,
		'adaptive_lasso_regressor':fit_adaptive_lasso_regressor,
		'gb_classifier':fit_gb_classifier,
		'gb_regressor':fit_gb_regressor,
		'logistic_regression':fit_l1_classifier,
		'lasso':fit_l1_regressor,
		'rf_classifier':fit_rf_classifier,
		'rf_regressor':fit_rf_regressor,
		'ufi_classifier':fit_ufi_classifier,
		'ufi_regressor':fit_ufi_regressor,
	}

	defaults = {
		'logistic_regression': {'penalty':'l1', 'solver':'liblinear', 'tol':1e-3, 'warm_start':True, 'class_weight':'balanced'},
		'gb_classifier': {'max_depth':1, 'colsample_bynode':1/3, 'n_estimators':100, 'importance_type':'gain'},
		'gb_regressor': {'max_depth':1, 'colsample_bynode':1/3, 'n_estimators':100, 'importance_type':'gain'},
		'rf_classifier': {'max_features':1/10, 'n_estimators':50},
		'rf_regressor': {'max_features':1/10, 'n_estimators':50},
		'ufi_classifier': {'n_estimators':100},
		'ufi_regressor': {'n_estimators':100},
	}

	selector_args = dict(selector_args or {})
	if selector in selectors:
		selector_function = selectors[selector]
		# user arguments are added to (and override) the defaults rather than replacing them
		selector_args = {**defaults.get(selector, {}), **selector_args}
	else:
		selector_function = selector
	return selector_function, selector_args


#----------------------------------------------------------------
# Stratified subsampling
#----------------------------------------------------------------
"""
Two disjoint subsamples, each of exactly subsample_size (at most n // 2), with the class proportions of labels.
Each class gets about 2 * subsample_size * (its share of the samples) places in total (largest remainders), and at
least 2 if it has 2 or more samples, so that it appears in both subsamples. Each class's places are split between
the subsamples as evenly as possible; odd totals alternate between the subsamples, so both have the same size.
rng is a RandomState (or the np.random module).
"""
def stratified_halves(labels, subsample_size, rng):
	labels = np.asarray(labels)
	n = labels.size
	if not is_integer(subsample_size) or not 1 <= subsample_size <= n // 2:
		raise ValueError(f'subsample_size must be an integer between 1 and n // 2 = {n // 2}, but got {subsample_size!r}.')
	classes = [np.flatnonzero(labels == c) for c in np.unique(labels)]
	counts = np.array([idx.size for idx in classes])
	total = 2 * subsample_size
	target = total * counts / n
	places = np.floor(target).astype(int)
	order = np.argsort(-(target - places), kind='stable')
	places[order[:total - places.sum()]] += 1
	for k in np.flatnonzero((places < 2) & (counts >= 2)):
		while places[k] < 2:
			donor = int(np.argmax(places))
			if places[donor] <= 2:
				break
			places[donor] -= 1
			places[k] += 1
	sizes = [[m // 2, m // 2] for m in places]
	extra = 0
	for k in np.flatnonzero(places % 2):
		sizes[k][extra] += 1
		extra = 1 - extra
	halves = [[], []]
	for idx, (a, b) in zip(classes, sizes):
		idx = rng.permutation(idx)
		halves[0].append(idx[:a])
		halves[1].append(idx[a:a + b])
	return [rng.permutation(np.concatenate(half)) for half in halves]

#----------------------------------------------------------------
# Jittering
#----------------------------------------------------------------
"""
"Just add noise" by Li and Melikechi (2026): tree importance scores favor continuous features over categorical ones,
because a categorical feature has only a few candidate splits. Adding a small amount of uniform noise to each
categorical feature, once before fitting, removes this bias. The noise for each column is Unif(-s, s) with
s = strength * (smallest gap between its distinct values), so integer-encoded categories get the paper's
Unif(-1e-4, 1e-4) and the categories stay separated for any encoding.
"""
jitter_max_levels = 10
jitter_strength = 1e-4

# columns with between 2 and max_levels distinct values
def categorical_columns(X, max_levels=jitter_max_levels):
	return np.array([j for j in range(X.shape[1]) if 2 <= np.unique(X[:, j]).size <= max_levels], dtype=int)

def jitter_columns(X, columns, rng, strength=jitter_strength):
	X = X.copy()
	for j in columns:
		values = np.unique(X[:, j])
		if values.size < 2:
			continue
		width = strength * np.min(np.diff(values))
		X[:, j] += rng.uniform(-width, width, X.shape[0])
	return X

#----------------------------------------------------------------
# Input validation
#----------------------------------------------------------------
def is_integer(value):
	return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

def is_number(value):
	return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) \
		and np.isfinite(value)

def validate_targets(target_fp, target_fdr):
	if target_fp is not None and target_fdr is not None:
		raise ValueError('Specify at most one of target_fp and target_fdr.')
	if target_fp is not None and not (is_number(target_fp) and target_fp > 0):
		raise ValueError(f'target_fp must be a positive number, but got {target_fp!r}.')
	if target_fdr is not None and not (is_number(target_fdr) and 0 < target_fdr <= 1):
		raise ValueError(f'target_fdr must be in (0, 1], but got {target_fdr!r}.')

def validate_scores(scores, shape, name):
	scores = np.asarray(scores, dtype=np.float64)
	if scores.shape != shape:
		raise ValueError(f'{name} returned an array of shape {scores.shape}; expected {shape}.')
	if not np.all(np.isfinite(scores)):
		raise ValueError(f'{name} returned non-finite values.')
	return scores
