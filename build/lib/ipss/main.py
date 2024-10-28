# Integrated path stability selection (IPSS)

import time
import warnings

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, lasso_path, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .base_selectors import fit_gb_classifier, fit_gb_regressor, fit_l1_classifier, fit_l1_regressor, fit_rf_classifier, fit_rf_regressor
from .helpers import check_response_type, compute_alphas, compute_qvalues, integrate, score_based_selection, selector_and_args
from .preselection import preselection

#--------------------------------
# IPSS
#--------------------------------
"""
Inputs:
	Required
	----------------
	X: n-by-p data matrix (n = number of samples, p = number of features)
	y: n-by-1 response vector (binary or continuous)

	Optional
	----------------
	selector: base algorithm; gradient boosting ('gb'), l1 regularized ('l1'), or random forest ('rf'')
	selector_args: arguments for base algorithm
	target_fp: target number of false positives
	target_fdr: target false discovery rate
	B: number of subsampling steps when computing selection probabilities
	n_alphas: number of values in grid of regularization or threshold parameters
	ipss_function: function to apply to selection probabilities; linear ('h1'), quadratic ('h2'), cubic ('h3')
	preselect: number (if int) or percentage (if 0 < preselect <= 1) of features to preselect. False for no preselection
	preselect_min: minimum number of features to keep in preselection step
	preselect_args: arguments for the preselection algorithm (see function called preselection)
	cutoff: max value of theoretical integral bound I(Lambda)
	delta: determines probability measure mu_delta(dlambda) = z_delta^{-1}lambda^{-delta}dlambda
	standardize_X: whether to standardize features to have mean 0, standard deviation 1
	center_y: whether to center the response to have mean 0
	true_features: list of true feature indices when they are known, e.g., in simulation experiments
	n_jobs: number of jobs to run in parallel

Outputs:
	efp_scores: efp (expected false positive) score for each feature
	q_values: q-value for each feature
	runtime: total runtime of the algorithm in seconds
	selected_features: the final set of selected features if target_fp or target_fdr is specified
	stability_paths: the stability paths for each feature (used for visualization)
"""
def ipss(X, y, selector='gb', selector_args=None, target_fp=None, target_fdr=None, B=None, n_alphas=None, ipss_function='h3', preselect=0.05, 
		preselect_min=200, preselector_args=None, cutoff=0.05, delta=1, standardize_X=None, center_y=None, true_features=None, n_jobs=1):

	# start timer
	start = time.time()

	# empty set for selector args if none specified
	selector_args = selector_args or {}

	# number of subsamples
	B = B if B is not None else 100 if selector == 'gb' else 50

	# reshape response
	if len(y.shape) > 1:
		y = y.ravel()
	
	# check response type
	binary_response, selector = check_response_type(y, selector)

	# standardize and center data if using l1 selectors
	if selector in ['lasso', 'logistic_regression']:
		if standardize_X is None:
			X = StandardScaler().fit_transform(X)
		if center_y is None:
			if not binary_response:
				y -= np.mean(y)

	# preselect features to reduce dimension
	if preselect:
		p_full = X.shape[1]
		X, preselect_indices, new_true = preselection(X, y, selector, preselect, preselect_min, preselector_args, true_features)
	else:
		preselect_indices, new_true = np.arange(X.shape[1]), true_features
	
	# dimensions post-preselection
	n, p = X.shape
	
	# maximum number of features for l1 regularized selectors (to avoid computational issues)
	max_features = 0.75 * p if selector in ['lasso', 'logistic_regression'] else None

	# alphas
	if not n_alphas:
		n_alphas = 100
	alphas = compute_alphas(X, y, n_alphas, max_features, binary_response) if selector in ['lasso', 'logistic_regression'] else None

	# selector function and args
	selector_function, selector_args = selector_and_args(selector, selector_args)

	# estimate selection probabilities
	results = np.array(Parallel(n_jobs=n_jobs)(delayed(selection)(X, y, alphas, selector_function, **selector_args) for _ in range(B)))

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

	# efp scores
	efp_scores = np.round(integral / np.maximum(scores, integral / p), decimals=8)
	efp_scores = sorted(zip(preselect_indices, efp_scores), key=lambda x: x[1])

	# reinsert features removed during preselection
	if preselect:
		kept_features = {efp_score[0] for efp_score in efp_scores}
		all_features = set(range(p_full))
		missing_features = all_features - kept_features
		for feature in missing_features:
			efp_scores.append((feature, p))
		efp_scores = [(feature, p_full if score >= p - 1 else score) for feature, score in efp_scores]

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
		selected_features = [feature for feature, score in efp_scores if score <= target_fp]
	else:
		selected_features = [feature for feature, q_value in q_values if q_value <= target_fdr]

	runtime = time.time() - start

	return { 
		'efp_scores': efp_scores,
		'q_values':q_values,
		'runtime': runtime, 
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

# subsampler for estimating selection probabilities
def selection(X, y, alphas, selector, **kwargs):
	n, p = X.shape
	indices = np.arange(n)
	np.random.shuffle(indices)
	n_split = int(len(indices) / 2)

	if alphas is None:
		indicators = np.empty((2,p))
		for half in range(2):
			idx = indices[:n_split] if half == 0 else indices[n_split:]
			indicators[half, :] = np.array(selector(X[idx,:], y[idx], **kwargs))
	else:
		indicators = np.empty((len(alphas), 2, p))
		for half in range(2):
			idx = indices[:n_split] if half == 0 else indices[n_split:]
			indicators[:, half, :] = np.array(selector(X[idx,:], y[idx], alphas, **kwargs))

	return indicators



