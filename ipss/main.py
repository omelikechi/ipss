# Integrated path stability selection

import time

from joblib import Parallel, delayed
import numpy as np

from .helpers import return_null_result
from .processing import postprocess_ipss, preprocess_ipss

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
	selector: gradient boosting ('gb'), l1 regularization ('l1'), random forest ('rf''), or a custom selector
	selector_args: arguments for selector
	target_fp: target number of false positives
	target_fdr: target false discovery rate
	B: number of subsampling steps when computing selection probabilities
	n_alphas: number of values in grid of regularization or threshold parameters
	ipss_function: function to apply to selection probabilities; linear ('h1'), quadratic ('h2'), cubic ('h3')
	preselect: number (if int) or percentage (if 0 < preselect <= 1) of features to preselect. False for no preselection
	preselect_args: arguments for the preselection algorithm (see function called preselection)
	cutoff: max value of theoretical integral bound I(Lambda)
	delta: determines probability measure mu_delta(dlambda) = z_delta^{-1}lambda^{-delta}dlambda
	subsample_size: size of the subsampled datasets
	standardize_X: whether to standardize features to have mean 0, standard deviation 1
	center_y: whether to center the response to have mean 0
	n_jobs: number of jobs to run in parallel
	force_regression: force ipss to use regression (if False, classification is used if response is binary)

Outputs:
	efp_scores: efp (expected false positive) score for each feature
	q_values: q-value for each feature
	runtime: total runtime of the algorithm in seconds
	selected_features: the final set of selected features if target_fp or target_fdr is specified
	stability_paths: the stability paths for each feature (used for visualization)
"""
def ipss(X, y, selector='gb', selector_args=None, preselect=True, preselector_args=None,
		B=None, n_alphas=None, ipss_function=None, cutoff=0.05, delta=None, subsample_size=None,
		target_fp=None, target_fdr=None, standardize_X=None, center_y=None, n_jobs=1,
		force_regression=False, _return_details=False):

	# start timer
	start = time.time()

	# prepare ipss arguments and data prior to estimating selection probabilities
	prep = preprocess_ipss(X, y, selector, selector_args, preselect, preselector_args,
		B, n_alphas, ipss_function, delta, standardize_X, center_y, force_regression)

	if prep is None:
		runtime = time.time() - start
		return return_null_result(X.shape[1], runtime)

	X, y = prep['X'], prep['y']
	selector_function, selector_args = prep['selector_function'], prep['selector_args']
	alphas, n_alphas, B, delta = prep['alphas'], prep['n_alphas'], prep['B'], prep['delta']
	p_full, preselect_indices = prep['p_full'], prep['preselect_indices']
	ipss_function = prep['ipss_function']

	n, p = X.shape

	# subsample size
	if subsample_size is None or subsample_size > n // 2:
		subsample_size = n // 2

	# estimate selection probabilities
	results = np.array(Parallel(n_jobs=n_jobs)(delayed(selection)(X, y, alphas, subsample_size,
		selector_function, **selector_args) for _ in range(B)))

	# process the estimated selection probabilities and compute final results (efp scores, q-values, etc.)
	ipss_output = postprocess_ipss(results, alphas, n_alphas, p, p_full, preselect_indices, B, cutoff, delta, 
		ipss_function, target_fdr, target_fp, _return_details)

	runtime = time.time() - start

	ipss_output['runtime'] = runtime

	return ipss_output

# subsampler for estimating selection probabilities
def selection(X, y, alphas, subsample_size, selector_function, **kwargs):
	n, p = X.shape
	indices = np.arange(n)
	np.random.shuffle(indices)

	# take two disjoint subsamples
	idx1 = indices[:subsample_size]
	idx2 = indices[subsample_size:2*subsample_size]

	if alphas is None:
		indicators = np.empty((2, p))
		indicators[0, :] = np.array(selector_function(X[idx1, :], y[idx1], **kwargs))
		indicators[1, :] = np.array(selector_function(X[idx2, :], y[idx2], **kwargs))

	else:
		indicators = np.empty((len(alphas), 2, p))
		indicators[:, 0, :] = np.array(selector_function(X[idx1, :], y[idx1], alphas, **kwargs))
		indicators[:, 1, :] = np.array(selector_function(X[idx2, :], y[idx2], alphas, **kwargs))

	return indicators


