# Integrated path stability selection

import time
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.utils import check_random_state

from .base_selectors import (fit_gb_classifier, fit_gb_regressor, fit_rf_classifier, fit_rf_regressor,
	fit_ufi_classifier, fit_ufi_regressor)
from .helpers import (is_integer, is_number, return_null_result, stratified_halves,
	validate_scores, validate_targets)
from .processing import postprocess_ipss, preprocess_ipss

# built-in selectors that take a random_state; each of their fits gets its own seed
seeded_selectors = (fit_gb_classifier, fit_gb_regressor, fit_rf_classifier, fit_rf_regressor,
	fit_ufi_classifier, fit_ufi_regressor)

#----------------------------------------------------------------
# IPSS
#----------------------------------------------------------------
"""
Inputs:
	Required
	----------------
	X: n-by-p data matrix (n = number of samples, p = number of features)
	y: n-by-1 response vector (binary or continuous)

	Optional
	----------------
	selector: gradient boosting ('gb'), l1 regularization ('l1'), random forest ('rf'), or a custom selector
	selector_args: arguments for selector
	target_fp: target number of false positives
	target_fdr: target false discovery rate
	B: number of subsampling steps when computing selection probabilities
	n_alphas: number of values in grid of regularization or threshold parameters
	ipss_function: function to apply to selection probabilities, h_m(x) = (2x - 1)^m for x > 0.5 (else 0), given as
		'h1', 'h2', 'h3', ... (default 'h2' for 'l1', 'h3' otherwise). Higher orders are more aggressive
	preselect: True to preselect features before subsampling with the preselector's default number of features,
		False for no preselection, an int k >= 1 to keep k features, or a float f in (0, 1] to keep the fraction f
		(rounded, at least one). For 'gb', True uses an adaptive rule (1.5 times the features with nonzero importance)
	preselector: preselection method; defaults to the matching selector for 'gb', 'l1', and 'adaptive_lasso', and to
		'dcor' (distance correlation) for every other selector. Also accepts 'gb', 'rf', 'l1', 'adaptive_lasso', or a custom preselector
		(see function called preselection)
	preselector_args: arguments for the preselection algorithm (see function called preselection)
	cutoff: max value of theoretical integral bound I(Lambda)
	delta: determines probability measure mu_delta(dlambda) = z_delta^{-1}lambda^{-delta}dlambda (default None, which sets
		delta from the feature correlations for 'l1' and uses 1 for every other selector; see compute_delta in helpers)
	subsample_size: size of each of the two disjoint subsamples (positive integer; default n // 2, and larger
		values are reduced to n // 2)
	stratify: for a binary response, draw each subsample with the same class proportions as y. This changes the
		sampling scheme (subsamples are no longer simple random samples), so it is off by default; without it,
		a subsample that contains only one class raises an error
	jitter: add a small amount of noise to categorical features once, before preselection and subsampling, so that
		tree importance scores do not favor continuous features (Li and Melikechi, "Just add noise", 2026). A feature
		is treated as categorical if it has 2 to 10 distinct values; each gets Unif(-s, s) noise, where s is 1e-4 times
		the smallest gap between its distinct values. None (default) jitters for the 'gb' and 'rf' selectors only,
		True for any selector, False never; a sequence of column indices jitters exactly those columns
	standardize_X: whether to standardize features to have mean 0, standard deviation 1 (None: only for 'l1' and 'adaptive_lasso')
	center_y: whether to center the response to have mean 0 (None: only for 'l1' and 'adaptive_lasso'; binary responses are never centered)
	n_jobs: number of jobs to run in parallel
	force_regression: force ipss to use regression (if False, classification is used if response is binary)
	random_state: seed (int, RandomState, or None) for preselection, subsampling, and the 'gb', 'rf', and 'ufi'
		selectors, each of whose fits gets its own seed. None uses numpy's global random state

Outputs:
	efp_scores: efp (expected false positive) score for each feature
	q_values: q-value for each feature
	runtime: total runtime of the algorithm in seconds
	selected_features: the final set of selected features if target_fp or target_fdr is specified
	stability_paths: the stability paths for each feature (used for visualization)
"""
def ipss(X, y, selector='gb', selector_args=None, preselect=True, preselector=None, preselector_args=None,
		B=None, n_alphas=None, ipss_function=None, cutoff=0.05, delta=None, subsample_size=None,
		target_fp=None, target_fdr=None, standardize_X=None, center_y=None, n_jobs=1,
		force_regression=False, random_state=None, stratify=False, jitter=None, _return_details=False):

	# start timer
	start = time.time()

	rng = check_random_state(random_state)

	# validate arguments that preprocessing does not check
	validate_targets(target_fp, target_fdr)
	if not is_number(cutoff) or cutoff <= 0:
		raise ValueError(f'cutoff must be a positive number, but got {cutoff!r}.')
	if subsample_size is not None and (not is_integer(subsample_size) or subsample_size < 1):
		raise ValueError(f'subsample_size must be a positive integer, but got {subsample_size!r}.')
	if delta is not None and not (np.ndim(delta) == 0 and is_number(np.asarray(delta).item())):
		raise ValueError(f'delta must be a single finite number, but got {delta!r}.')
	if not isinstance(stratify, (bool, np.bool_)):
		raise ValueError(f'stratify must be True or False, but got {stratify!r}.')

	# built-in selectors get their own seed for every fit, so a seed in selector_args (which would give every
	# fit the same one) is dropped here, before defaults are filled in; if it was the only argument, the
	# selector's defaults still apply
	if isinstance(selector, str) and isinstance(selector_args, dict) and 'random_state' in selector_args:
		warnings.warn("selector_args['random_state'] is ignored so that every fit gets its own seed; "
			'use random_state instead.', UserWarning)
		selector_args = {key: value for key, value in selector_args.items() if key != 'random_state'} or None

	# prepare ipss arguments and data prior to estimating selection probabilities
	prep = preprocess_ipss(X, y, selector, selector_args, preselect, preselector, preselector_args,
		B, n_alphas, ipss_function, delta, standardize_X, center_y, force_regression, rng, jitter)

	if prep is None:
		runtime = time.time() - start
		return return_null_result(np.shape(X)[1], runtime)

	X, y = prep['X'], prep['y']
	selector_function, selector_args = prep['selector_function'], prep['selector_args']
	alphas, n_alphas, B, delta = prep['alphas'], prep['n_alphas'], prep['B'], prep['delta']
	p_full, preselect_indices = prep['p_full'], prep['preselect_indices']
	ipss_function = prep['ipss_function']
	binary_response = prep['binary_response']

	n, p = X.shape

	# subsample size
	if subsample_size is None or subsample_size > n // 2:
		subsample_size = n // 2

	# one seed per subsampling step
	seeds = rng.randint(np.iinfo(np.int32).max, size=B)

	# estimate selection probabilities
	results = np.array(Parallel(n_jobs=n_jobs)(delayed(selection)(X, y, alphas, subsample_size,
		selector_function, seeds[b], stratify and binary_response, binary_response, **selector_args) for b in range(B)))

	# process the estimated selection probabilities and compute final results (efp scores, q-values, etc.)
	ipss_output = postprocess_ipss(results, alphas, n_alphas, p, p_full, preselect_indices, B, cutoff, delta, 
		ipss_function, target_fdr, target_fp, _return_details)

	runtime = time.time() - start

	ipss_output['runtime'] = runtime

	return ipss_output


# subsampler for estimating selection probabilities
def selection(X, y, alphas, subsample_size, selector_function, seed=None, stratify=False, binary=False, **kwargs):
	n, p = X.shape
	rng = check_random_state(seed)

	# take two disjoint subsamples
	if stratify:
		if np.min(np.unique(y, return_counts=True)[1]) < 2:
			raise ValueError('stratify=True needs at least 2 samples of each class.')
		halves = stratified_halves(y, subsample_size, rng)
	else:
		indices = rng.permutation(n)
		halves = [indices[:subsample_size], indices[subsample_size:2*subsample_size]]
	if binary:
		for idx in halves:
			if np.unique(y[idx]).size < 2:
				raise ValueError('A subsample contains only one class of the binary response. Use stratify=True to '
					'keep both classes in every subsample.')
	alpha_args = () if alphas is None else (alphas,)
	shape = (p,) if alphas is None else (len(alphas), p)

	# built-in tree selectors get a new seed for each fit unless one is given
	seed_fits = selector_function in seeded_selectors and 'random_state' not in kwargs

	indicators = np.empty((2, p)) if alphas is None else np.empty((len(alphas), 2, p))

	# seed numpy's global random state too, so that custom selectors that draw from it are reproducible
	# for any n_jobs (each parallel worker otherwise has its own unseeded state); restored afterward
	state = np.random.get_state()
	np.random.seed(rng.randint(np.iinfo(np.int32).max))
	try:
		for i, idx in enumerate(halves):
			fit_args = {**kwargs, 'random_state': rng.randint(np.iinfo(np.int32).max)} if seed_fits else kwargs
			output = selector_function(X[idx, :], y[idx], *alpha_args, **fit_args)
			indicators[..., i, :] = validate_scores(output, shape, 'the selector')
	finally:
		np.random.set_state(state)

	return indicators
