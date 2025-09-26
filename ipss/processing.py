# Functions for processing IPSS before (pre) and after (post) estimating selection probabilities

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

from .helpers import (check_response_type, compute_alphas, compute_delta, compute_qvalues, integrate, 
	return_null_result, score_based_selection, selector_and_args)
from .preselection import preselection

# prepare ipss arguments and data
def preprocess_ipss(X, y, selector, selector_args, preselect, preselector_args, 
	B, n_alphas, ipss_function, delta, standardize_X, center_y):

	# specify whether base estimator is a regularization or variable importance method
	selector_type = 'importance'
	if selector in ['adaptive_lasso', 'l1']:
		selector_type = 'regularization'

	# empty set for selector args if none specified
	selector_args = selector_args or {}

	# add selector_args to preselector_args for custom selectors
	if not isinstance(selector, str):
		if preselector_args is None:
			preselector_args = {}
		for key in selector_args:
			if key not in preselector_args:
				preselector_args[key] = selector_args[key]

	# number of subsamples
	B = B if B is not None else 100 if selector == 'gb' else 50

	# reshape response
	if len(y.shape) > 1:
		y = y.ravel()
	
	# check response type
	binary_response, selector = check_response_type(y, selector)

	# ipss function
	if ipss_function is None:
		ipss_function = 'h2' if selector in ['lasso', 'logistic_regression'] else 'h3'

	# probability measure
	if delta is None:
		delta = compute_delta(X, selector)

	# standardize and center data if using l1 selectors
	if selector_type == 'regularization':
		if standardize_X is None:
			X = StandardScaler().fit_transform(X)
		if center_y is None:
			if not binary_response:
				y -= np.mean(y)

	# preselect features to reduce dimension
	p_full = X.shape[1]
	if preselect:
		X, preselect_indices = preselection(X, y, selector, preselector_args, selector_type)
		if preselect_indices.size == 0:
			warnings.warn('Preselection step removed all features. Returning null result.', UserWarning)
			return None
	else:
		preselect_indices = np.arange(p_full)
	
	# dimensions post-preselection
	n, p = X.shape
	
	# maximum number of features for l1 regularized selectors (to avoid computational issues)
	max_features = 0.75 * p if selector_type == 'regularization' else None

	# alphas
	if n_alphas is None:
		n_alphas = 25 if selector_type == 'regularization' else 100
	alphas = compute_alphas(X, y, n_alphas, max_features, binary_response) if selector_type == 'regularization' else None

	# selector function and args
	selector_function, selector_args = selector_and_args(selector, selector_args)

	return {
		'X':X,
		'y':y,
		'selector_function':selector_function,
		'selector_args':selector_args,
		'alphas':alphas,
		'n_alphas':n_alphas,
		'B':B,
		'delta':delta,
		'ipss_function':ipss_function,
		'p_full':p_full,
		'preselect_indices':preselect_indices
	}

# postprocess estimated selection probabilities
def postprocess_ipss(results, alphas, n_alphas, p, p_full, preselect_indices, B, cutoff, delta, 
	ipss_function, target_fdr, target_fp):

	# score-based selection
	if alphas is None:
		results, alphas = score_based_selection(results, n_alphas)

	# aggregate results
	Z = np.zeros((n_alphas, 2*B, p))
	for b in range(B):
		Z[:, 2*b:2*(b + 1), :] = results[b,:,:,:]

	# average number of features selected (denoted q in ipss papers)
	average_selected = np.array([np.mean(np.sum(Z[i,:,:], axis=1)) for i in range(n_alphas)])

	# stability paths
	stability_paths = np.empty((n_alphas,p))
	for i in range(n_alphas):
		stability_paths[i] = Z[i].mean(axis=0)

	# stop if all stability paths stop changing (after burn-in period where mean selection probability < 0.01)
	stop_index = n_alphas
	for i in range(2,len(alphas)):
		if np.isclose(stability_paths[i,:], np.zeros(p)).all():
			continue
		else:
			diff = stability_paths[i,:] - stability_paths[i-2,:]
			mean = np.mean(stability_paths[i,:])
			if np.isclose(diff, np.zeros(p)).all() and mean > 0.01:
				stop_index = i
				break

	# truncate stability paths at stop index
	stability_paths = stability_paths[:stop_index,:]
	alphas = alphas[:stop_index]
	average_selected = average_selected[:stop_index]

	# compute feature-specific ipss integral scores and false positive bound
	scores, integral, alphas, stop_index = ipss_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff)

	efp_scores = np.round(integral / np.maximum(scores, integral / p), decimals=8)
	efp_scores = dict(zip(map(int, preselect_indices), efp_scores))

	# reinsert features removed during preselection
	if p_full != p:
		all_features = set(range(p_full))
		missing_features = all_features - efp_scores.keys()
		for feature in missing_features:
			efp_scores[feature] = p
		efp_scores = {feature: (p_full if score >= p - 1 else score) for feature, score in efp_scores.items()}

	# reindex stability paths based on original features
	stability_paths_full = np.zeros((stability_paths.shape[0], p_full))
	stability_paths_full[:, preselect_indices] = stability_paths
	stability_paths = stability_paths_full

	# q_values
	q_values = compute_qvalues(efp_scores)

	# select features if either target_fp or target_fdr is specified
	if not target_fp and not target_fdr:
		selected_features = []
	elif target_fp:
		selected_features = [feature for feature, efp_score in efp_scores.items() if efp_score <= target_fp]
	else:
		selected_features = [feature for feature, q_value in q_values.items() if q_value <= target_fdr]

	return { 
		'efp_scores': efp_scores,
		'q_values':q_values,
		'selected_features': selected_features, 
		'stability_paths': stability_paths
		}


# compute ipss scores and theoretical E(FP) bounds
def ipss_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff):
	n_alphas, p = stability_paths.shape

	if ipss_function not in ['h1', 'h2', 'h3']:
		raise ValueError(f"ipss_function must be 'h1', 'h2', or 'h3', but got ipss_function = {ipss_function} instead")

	m = 1 if ipss_function == 'h1' else 2 if ipss_function == 'h2' else 3

	# function to apply to selection probabilities
	def h_m(x):
		return 0 if x <= 0.5 else (2*x - 1)**m

	# evaluate ipss bounds for specific functions
	if m == 1:
		integral, stop_index = integrate(average_selected**2 / p, alphas, delta, cutoff=cutoff)
	elif m == 2:
		term1 = average_selected**2 / (p * B)
		term2 = (B-1) * average_selected**4 / (B * p**3)
		integral, stop_index  = integrate(term1 + term2, alphas, delta, cutoff=cutoff)
	else:
		term1 = average_selected**2 / (p * B**2)
		term2 = (3 * (B-1) * average_selected**4) / (p**3 * B**2)
		term3 = ((B-1) * (B-2) * average_selected**6) / (p**5 * B**2)
		integral, stop_index = integrate(term1 + term2 + term3, alphas, delta, cutoff=cutoff)

	# compute ipss scores
	alphas_stop = alphas[:stop_index]
	scores = np.zeros(p)
	for i in range(p):
		values = np.empty(stop_index)
		for j in range(stop_index):
			values[j] = h_m(stability_paths[j,i])
		scores[i], _ = integrate(values, alphas_stop, delta)

	return scores, integral, alphas, stop_index


