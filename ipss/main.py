# Integrated path stability selection

import time
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.utils import check_random_state

from .base_selectors import (fit_gb_classifier, fit_gb_regressor, fit_rf_classifier, fit_rf_regressor,
	fit_ufi_classifier, fit_ufi_regressor)
from .helpers import return_null_result
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
	ipss_function: function to apply to selection probabilities; linear ('h1'), quadratic ('h2'), cubic ('h3')
	preselect: number (if int) or percentage (if 0 < preselect <= 1) of features to preselect. False for no preselection
	preselector: preselection method; defaults to 'dcor' (distance correlation) for every selector except 'l1'/'adaptive_lasso',
		which default to matching selector. Also accepts 'gb', 'rf', 'ufi', 'l1', 'adaptive_lasso', or a custom preselector
		(see function called preselection)
	preselector_args: arguments for the preselection algorithm (see function called preselection)
	cutoff: max value of theoretical integral bound I(Lambda)
	delta: determines probability measure mu_delta(dlambda) = z_delta^{-1}lambda^{-delta}dlambda
	subsample_size: size of the subsampled datasets
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
		force_regression=False, random_state=None, _return_details=False):

	# start timer
	start = time.time()

	rng = check_random_state(random_state)

	# prepare ipss arguments and data prior to estimating selection probabilities
	prep = preprocess_ipss(X, y, selector, selector_args, preselect, preselector, preselector_args,
		B, n_alphas, ipss_function, delta, standardize_X, center_y, force_regression, rng)

	if prep is None:
		runtime = time.time() - start
		return return_null_result(np.shape(X)[1], runtime)

	X, y = prep['X'], prep['y']
	selector_function, selector_args = prep['selector_function'], prep['selector_args']
	alphas, n_alphas, B, delta = prep['alphas'], prep['n_alphas'], prep['B'], prep['delta']
	p_full, preselect_indices = prep['p_full'], prep['preselect_indices']
	ipss_function = prep['ipss_function']

	if selector_function in seeded_selectors and 'random_state' in selector_args:
		warnings.warn("selector_args['random_state'] is ignored; use random_state instead.", UserWarning)
		selector_args = {key: value for key, value in selector_args.items() if key != 'random_state'}

	n, p = X.shape

	# subsample size
	if subsample_size is None or subsample_size > n // 2:
		subsample_size = n // 2

	# one seed per subsampling step
	seeds = rng.randint(np.iinfo(np.int32).max, size=B)

	# estimate selection probabilities
	results = np.array(Parallel(n_jobs=n_jobs)(delayed(selection)(X, y, alphas, subsample_size,
		selector_function, seeds[b], **selector_args) for b in range(B)))

	# process the estimated selection probabilities and compute final results (efp scores, q-values, etc.)
	ipss_output = postprocess_ipss(results, alphas, n_alphas, p, p_full, preselect_indices, B, cutoff, delta, 
		ipss_function, target_fdr, target_fp, _return_details)

	runtime = time.time() - start

	ipss_output['runtime'] = runtime

	return ipss_output


# subsampler for estimating selection probabilities
def selection(X, y, alphas, subsample_size, selector_function, seed=None, **kwargs):
	n, p = X.shape
	rng = check_random_state(seed)
	indices = rng.permutation(n)

	# take two disjoint subsamples
	halves = [indices[:subsample_size], indices[subsample_size:2*subsample_size]]
	alpha_args = () if alphas is None else (alphas,)

	# built-in tree selectors get a new seed for each fit unless one is given
	seed_fits = selector_function in seeded_selectors and 'random_state' not in kwargs

	indicators = np.empty((2, p)) if alphas is None else np.empty((len(alphas), 2, p))
	for i, idx in enumerate(halves):
		fit_args = {**kwargs, 'random_state': rng.randint(np.iinfo(np.int32).max)} if seed_fits else kwargs
		indicators[..., i, :] = selector_function(X[idx, :], y[idx], *alpha_args, **fit_args)

	return indicators


