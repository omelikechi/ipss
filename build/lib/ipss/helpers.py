# Helper functions for IPSS

import warnings

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression

from .base_selectors import fit_gb_classifier, fit_gb_regressor, fit_l1_classifier, fit_l1_regressor, fit_rf_classifier, fit_rf_regressor

def check_response_type(y, selector):
	unique_values = np.unique(y)
	if len(unique_values) == 1:
		print(f"Error: The response variable `y` has only one unique value: {unique_values[0]}.")
		return None, None
	binary_response = len(unique_values) == 2
	if binary_response:
		minval = np.min(unique_values)
		y = np.where(y == minval, 0, 1)
	if selector == 'l1':
		selector = 'logistic_regression' if binary_response else 'lasso'
	if selector == 'rf':
		selector = 'rf_classifier' if binary_response else 'rf_regressor'
	elif selector == 'gb':
		selector = 'gb_classifier' if binary_response else 'gb_regressor'
	return binary_response, selector

def compute_alphas(X, y, n_alphas, max_features, binary_response=False):
	n, p = X.shape
	if binary_response:
		y_mean = np.mean(y)
		scaled_residuals = y - y_mean * (1 - y_mean)
		alpha_max = 5 / np.max(np.abs(np.dot(X.T, scaled_residuals) / n))
		selector = LogisticRegression(penalty='l1', solver='saga', tol=1e-3, warm_start=True, class_weight='balanced')
		# selector = LogisticRegression(penalty='l1', max_iter=int(1e6), solver='liblinear', class_weight='balanced')
		if np.isnan(alpha_max):
			alpha_max = 100
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

def compute_qvalues(efp_scores):
	T = [efp_score for _, efp_score in efp_scores]
	fdrs = []
	for t in T:
		efp_scores_leq_t = [efp_score for _, efp_score in efp_scores if efp_score <= t]
		FP = max(efp_scores_leq_t)
		S = len(efp_scores_leq_t)
		fdrs.append((t, FP/S))
	q_values = []
	for feature, score in efp_scores:
		q_value = min([fdr for t, fdr in fdrs if score <= t])
		q_values.append((feature, q_value))
	return q_values

def integrate(values, alphas, delta=1, cutoff=None):
	n_alphas = len(alphas)
	a = min(alphas)
	b = max(alphas)
	if delta == 1:
		normalization = (1 - (a/b)**(1/n_alphas)) / np.log(b/a)
	else:
		normalization = (1 - delta) * (1 - (a/b)**(1/n_alphas)) / (b**(1-delta) - a**(1-delta))
	output = 0
	stop_index = n_alphas
	before = stop_index
	if cutoff is None:
		for i in range(1,n_alphas):
			weight = 1 if delta == 1 else alphas[i]**(1-delta)
			output += normalization * weight * values[i-1]
	else:
		for i in range(1,n_alphas):
			weight = 1 if delta == 1 else alphas[i]**(1-delta)
			updated_output = output + normalization * weight * values[i-1]
			if updated_output > cutoff:
				stop_index = i
				break
			else:
				output = updated_output
	return output, stop_index

def score_based_selection(results, n_alphas):
	n_alphas = max(100, n_alphas)	
	alpha_min = np.min(results)
	if alpha_min < 0:
		results += np.abs(alpha_min)
	alpha_max = np.max(results) + .01
	alpha_min = alpha_max / 1e8
	alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
	B, _, p = results.shape

	reshape_results = np.empty((B, n_alphas, 2, p))
	for i in range(n_alphas):
		reshape_results[:,i,:,:] = results
	results = reshape_results
	for i, alpha in enumerate(alphas):
		for b in range(B):
			for j in range(2):
				results[b,i,j,:] = (results[b,i,j,:] > alpha).astype(int)
	return results, alphas

def selector_and_args(selector, selector_args):
	selectors = {'gb_classifier':fit_gb_classifier, 'gb_regressor':fit_gb_regressor, 'logistic_regression':fit_l1_classifier,
		'lasso':fit_l1_regressor, 'rf_classifier':fit_rf_classifier, 'rf_regressor':fit_rf_regressor}
	if selector in selectors:
		selector_function = selectors[selector]
		if selector == 'logistic_regression' and not selector_args:
			selector_args = {'penalty': 'l1', 'solver': 'saga', 'tol': 1e-3, 'warm_start': True, 'class_weight': 'balanced'}
		elif selector in ['gb_classifier', 'gb_regressor'] and not selector_args:
			selector_args = {'max_depth':1, 'colsample_bynode':1/3, 'n_estimators':100, 'importance_type':'gain'}
		elif selector in ['rf_classifier', 'rf_regressor'] and not selector_args:
			selector_args = {'max_features':1/3, 'n_estimators':100}
	else:
		selector_function = selector
	return selector_function, selector_args