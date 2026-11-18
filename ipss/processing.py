# Functions for processing IPSS before (pre) and after (post) estimating selection probabilities

import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from .base_selectors import adaptive_weights
from .helpers import (categorical_columns, check_response_type, compute_alphas, compute_qvalues, default_delta,
	efp_bound_integrand, integrate, ipss_function_order, is_integer, is_number, jitter_columns, resolve_preselector,
	resolve_selector, score_based_selection, selector_and_args, validate_targets)
from .preselection import preselection

valid_selectors = ('gb', 'l1', 'rf', 'adaptive_lasso', 'ufi')

# internal selector names, accepted as aliases of the public ones
selector_aliases = {'gb_classifier': 'gb', 'gb_regressor': 'gb', 'lasso': 'l1', 'logistic_regression': 'l1',
	'rf_classifier': 'rf', 'rf_regressor': 'rf', 'adaptive_lasso_classifier': 'adaptive_lasso',
	'adaptive_lasso_regressor': 'adaptive_lasso', 'ufi_classifier': 'ufi', 'ufi_regressor': 'ufi'}

# prepare ipss arguments and data
def preprocess_ipss(X, y, selector, selector_args, preselect, preselector, preselector_args,
	B, n_alphas, ipss_function, delta, standardize_X, center_y, force_regression=False, random_state=None,
	jitter=None):

	random_state = check_random_state(random_state)

	# cast to float64 (float32 alpha grids underflow/overflow in integrate)
	X = X.astype(np.float64) if issparse(X) else np.asarray(X, dtype=np.float64)
	y = np.asarray(y)
	if y.dtype.kind in 'biuf':
		y = y.astype(np.float64)

	# validate data and arguments
	if X.ndim != 2:
		raise ValueError(f'X must be two-dimensional, but has shape {X.shape}.')
	if y.ndim > 2 or (y.ndim == 2 and 1 not in y.shape):
		raise ValueError(f'y must be one-dimensional, but has shape {y.shape}.')
	if y.size != X.shape[0]:
		raise ValueError(f'X has {X.shape[0]} samples but y has {y.size}.')
	if X.shape[0] < 2 or X.shape[1] < 2:
		raise ValueError(f'X must have at least 2 samples and 2 features, but has shape {X.shape}.')
	if not np.all(np.isfinite(X.data if issparse(X) else X)):
		raise ValueError('X contains missing or non-finite values.')
	if y.dtype.kind in 'biuf' and not np.all(np.isfinite(y)):
		raise ValueError('y contains missing or non-finite values.')
	# preselect: True/False, a number of features to keep (int >= 1), or a fraction of features to keep (float in (0, 1])
	if not isinstance(preselect, (bool, np.bool_)):
		if is_integer(preselect) and preselect >= 1:
			n_keep = int(min(preselect, X.shape[1]))
		elif isinstance(preselect, (float, np.floating)) and np.isfinite(preselect) and 0 < preselect <= 1:
			n_keep = max(1, int(round(preselect * X.shape[1])))
		else:
			raise ValueError(f'preselect must be True, False, a positive integer (number of features to keep), or a '
				f'float in (0, 1] (fraction of features to keep), but got {preselect!r}.')
		if preselector_args and 'n_keep' in preselector_args:
			raise ValueError("Specify the number of preselected features with preselect or preselector_args['n_keep'], "
				'not both.')
		preselector_args = {**(preselector_args or {}), 'n_keep': n_keep}
		preselect = True
	if B is not None and (not is_integer(B) or B < 1):
		raise ValueError(f'B must be a positive integer, but got {B!r}.')
	if n_alphas is not None and (not is_integer(n_alphas) or n_alphas < 2):
		raise ValueError(f'n_alphas must be an integer of at least 2, but got {n_alphas!r}.')
	# a sequence of deltas is allowed for callers that postprocess once per delta (e.g. causalstabsel)
	if delta is not None:
		if np.ndim(delta) == 0:
			delta = np.asarray(delta).item()
			valid = is_number(delta)
		else:
			delta = [np.asarray(d).item() for d in np.ravel(delta)]
			valid = len(delta) > 0 and all(is_number(d) for d in delta)
		if not valid:
			raise ValueError(f'delta must be a finite number (or a sequence of them), but got {delta!r}.')
	if isinstance(selector, str):
		selector = selector_aliases.get(selector, selector)
	if not callable(selector) and selector not in valid_selectors:
		raise ValueError(f"Unrecognized selector: {selector!r}. Must be one of 'gb', 'l1', 'rf', 'adaptive_lasso', "
			"'ufi', or a custom callable.")
	if ipss_function is not None:
		ipss_function_order(ipss_function)
	for name, args in (('selector_args', selector_args), ('preselector_args', preselector_args)):
		if args is not None and not isinstance(args, dict):
			raise ValueError(f'{name} must be a dict or None, but got {args!r}.')
	for name, value in (('standardize_X', standardize_X), ('center_y', center_y)):
		if value is not None and not isinstance(value, (bool, np.bool_)):
			raise ValueError(f'{name} must be True, False, or None, but got {value!r}.')

	# jitter categorical features (see helpers): None jitters them only for the tree selectors 'gb' and 'rf',
	# True for any selector, False never; a sequence of column indices jitters exactly those columns
	if jitter is None:
		jitter = selector in ('gb', 'rf') and not issparse(X)
	if isinstance(jitter, (bool, np.bool_)):
		if jitter and issparse(X):
			raise ValueError('jitter is not supported for sparse X.')
		jitter_indices = categorical_columns(X) if jitter else np.array([], dtype=int)
	else:
		jitter_indices = np.asarray(jitter)
		if jitter_indices.ndim != 1 or not all(is_integer(j) and 0 <= j < X.shape[1] for j in jitter_indices.tolist()):
			raise ValueError(f'jitter must be None, True, False, or a sequence of column indices, but got {jitter!r}.')
		if issparse(X) and jitter_indices.size:
			raise ValueError('jitter is not supported for sparse X.')
		jitter_indices = np.unique(jitter_indices.astype(int))
	if jitter_indices.size:
		X = jitter_columns(X, jitter_indices, random_state)

	# specify whether base estimator is a regularization or variable importance method
	selector_type = 'importance'
	if selector in ['adaptive_lasso', 'l1']:
		selector_type = 'regularization'

	# empty set for selector args if none specified
	selector_args = selector_args or {}

	# number of subsamples
	B = B if B is not None else 100 if selector == 'gb' else 50

	# reshape response
	if len(y.shape) > 1:
		y = y.ravel()
	
	# check response type
	if np.unique(y).size == 1:
		raise ValueError(f'The response y has only one unique value: {y[0]}.')
	if force_regression:
		binary_response = False
		selector = resolve_selector(selector, binary_response)
	else:
		binary_response, selector = check_response_type(y, selector)

	# code binary responses as 0/1 (required by XGBoost)
	if binary_response:
		y = (y == np.unique(y)[1]).astype(np.float64)

	# ipss function
	if ipss_function is None:
		ipss_function = 'h2' if selector in ['lasso', 'logistic_regression'] else 'h3'

	# probability measure
	if delta is None:
		delta = default_delta

	# standardize and center data (by default only for l1 selectors); never center a binary response
	if standardize_X is None:
		standardize_X = selector_type == 'regularization'
	if center_y is None:
		center_y = selector_type == 'regularization'
	if standardize_X:
		X = StandardScaler().fit_transform(X)
	if center_y and not binary_response:
		y = y - np.mean(y)

	# preselect features to reduce dimension
	p_full = X.shape[1]
	if preselect:
		preselector = resolve_preselector(preselector, selector, binary_response)
		X, preselect_indices = preselection(X, y, preselector, preselector_args, random_state)
		if preselect_indices.size == 0:
			warnings.warn('Preselection step removed all features. Returning null result.', UserWarning)
			return None
	else:
		preselect_indices = np.arange(p_full)
	
	# dimensions post-preselection
	n, p = X.shape

	# no feature can be selected by an l1 selector if every feature is constant (the alpha grid is undefined)
	if selector_type == 'regularization' and np.all(np.ptp(X.toarray() if issparse(X) else X, axis=0) == 0):
		warnings.warn('Every feature is constant, so no feature can be selected. Returning null result.', UserWarning)
		return None
	
	# no feature can be selected by an l1 selector if no feature is correlated with a continuous response
	if selector_type == 'regularization' and not binary_response and not issparse(X) and np.max(np.abs(X.T @ y)) == 0:
		warnings.warn('No feature is correlated with the response, so no feature can be selected. Returning null '
			'result.', UserWarning)
		return None

	# maximum number of features for l1 regularized selectors (to avoid computational issues)
	max_features = 0.75 * p if selector_type == 'regularization' else None

	# alphas
	if n_alphas is None:
		n_alphas = 25 if selector_type == 'regularization' else 100
	alphas = None
	if selector_type == 'regularization':
		# adaptive lasso fits a lasso on X / weights, so calibrate the grid on that reweighted design
		X_alphas = X
		if selector in ['adaptive_lasso_classifier', 'adaptive_lasso_regressor']:
			epsilon = selector_args.get('epsilon', 1e-6)
			X_alphas = X / adaptive_weights(X, y, binary_response, epsilon)
		alphas = compute_alphas(X_alphas, y, n_alphas, max_features, binary_response)

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
		'preselect_indices':preselect_indices,
		'binary_response':binary_response,
		'jitter_indices':jitter_indices
	}

# postprocess estimated selection probabilities
def postprocess_ipss(results, alphas, n_alphas, p, p_full, preselect_indices, B, cutoff, delta, 
	ipss_function, target_fdr, target_fp, _return_details=False):

	validate_targets(target_fp, target_fdr)

	# score-based selection
	if alphas is None:
		results, alphas = score_based_selection(results, n_alphas)
	alphas = np.asarray(alphas, dtype=np.float64)

	# aggregate results
	Z = np.zeros((n_alphas, 2*B, p))
	for b in range(B):
		Z[:, 2*b:2*(b + 1), :] = results[b,:,:,:]

	# average number of features selected (denoted q in ipss papers)
	average_selected = Z.sum(axis=2).mean(axis=1)

	# stability paths
	stability_paths = Z.mean(axis=1)

	# drop the final plateau, where no stability path changes for the rest of the grid, keeping its first point
	# (all fits are already computed, so only a tail that is constant through the end of the grid is removed)
	changed = np.flatnonzero(np.any(stability_paths != stability_paths[-1], axis=1))
	plateau_start = changed[-1] + 1 if changed.size > 0 else 0
	keep = min(n_alphas, plateau_start + 2)
	stability_paths, alphas, average_selected, Z = stability_paths[:keep], alphas[:keep], average_selected[:keep], Z[:keep]

	# compute feature-specific ipss integral scores and false positive bound on the interval chosen by the cutoff
	scores, integral, stop_index = ipss_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff)
	stability_paths, alphas, average_selected, Z = (stability_paths[:stop_index], alphas[:stop_index],
		average_selected[:stop_index], Z[:stop_index])

	# efp(j) = min(I / integral of h_m(pi_j), p); features with a zero integral are never selected (efp = p)
	selected = scores > 0
	efp = np.full(p, float(p))
	efp[selected] = np.minimum(integral / scores[selected], p)
	efp = np.round(efp, decimals=8)

	# expand to the original features: those removed during preselection, and retained ones with efp = p, get p_full
	efp_full = np.full(p_full, float(p_full))
	efp_full[preselect_indices] = np.where(efp >= p, p_full, efp)
	efp_scores = {j: float(score) for j, score in enumerate(efp_full)}

	# reindex stability paths based on original features
	stability_paths_full = np.zeros((stability_paths.shape[0], p_full))
	stability_paths_full[:, preselect_indices] = stability_paths
	stability_paths = stability_paths_full

	# q_values
	q_values = compute_qvalues(efp_scores)

	# select features if either target_fp or target_fdr is specified
	if target_fp is None and target_fdr is None:
		selected_features = []
	elif target_fp is not None:
		selected_features = [feature for feature, efp_score in efp_scores.items() if efp_score <= target_fp]
	else:
		selected_features = [feature for feature, q_value in q_values.items() if q_value <= target_fdr]

	result = {'efp_scores':efp_scores,'q_values':q_values,'selected_features':selected_features,'stability_paths':stability_paths}

	if _return_details:
		m = ipss_function_order(ipss_function)
		result.update({'alphas':alphas, 'average_selected':average_selected, 'delta':delta, 'indicators':Z, 
			'integral':integral, 'ipss_function_order':m, 'p_full':p_full, 'preselect_indices':preselect_indices})

	return result

# compute ipss scores and theoretical E(FP) bound, both integrated over the same interval with the same measure
def ipss_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff):
	n_alphas, p = stability_paths.shape

	m = ipss_function_order(ipss_function)

	# integrand of the E(FP) bound I(Lambda) (Theorem 4.1 for m = 1, 2, 3; see efp_bound_integrand)
	bound = efp_bound_integrand(average_selected, p, B, m)

	# lambda_min from the cutoff; I(Lambda) is normalized on [lambda_min, lambda_max]
	integral, stop_index = integrate(bound, alphas, delta, cutoff=cutoff)

	# h_m(x) = (2x - 1)^m for x > 0.5, else 0, integrated over the same interval with the same weights
	paths = stability_paths[:stop_index]
	h = np.where(paths > 0.5, (2*paths - 1)**m, 0.0)
	scores, _ = integrate(h, alphas[:stop_index], delta)

	return scores, integral, stop_index
