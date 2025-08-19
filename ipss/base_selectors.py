# Base estimators for IPSS
"""
Baseline selectors based on:
	- Importance scores from gradient boosting with XGBoost
	- l1-regularized linear (lasso) or logistic regression
	- Importance scores from random forests with scikit learn
"""

import warnings

import numpy as np
from skglm.estimators import GeneralizedLinearEstimator, MCPRegression
from skglm.penalties import SCAD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, lasso_path, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

# adaptive lasso classifier
def fit_adaptive_lasso_classifier(X, y, alphas, epsilon=1e-6):
	n_alphas = len(alphas)
	n_features = X.shape[1]
	coefficients = np.zeros((n_alphas, n_features))

	# Initial logistic regression with L2 penalty
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		init_model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
		init_model.fit(X, y)
		init_coef = init_model.coef_.flatten()

	weights = 1 / (np.abs(init_coef) + epsilon)

	for i, alpha in enumerate(alphas):
		X_weighted = X / weights[np.newaxis, :]
		model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear', max_iter=1000)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			model.fit(X_weighted, y)
		coef = model.coef_.flatten() / weights
		coefficients[i, :] = (coef != 0).astype(int)
	return coefficients





# def fit_adaptive_lasso_classifier(X, y, alphas, epsilon=1e-6):
# 	n_alphas = len(alphas)
# 	n_features = X.shape[1]
# 	coefficients = np.zeros((n_alphas, n_features))

# 	# get initial coefficients from ridge regression
# 	with warnings.catch_warnings():
# 		warnings.simplefilter('ignore')
# 		init_ridge = RidgeClassifier()
# 		init_ridge.fit(X,y)
# 		init_coef = init_ridge.coef_.flatten()

# 	# compute adaptive weights
# 	weights = 1 / (np.abs(init_coef) + epsilon)

# 	# Solve weighted l1-penalized logistic regression
# 	for i, alpha in enumerate(alphas):
# 		X_weighted = X / weights[np.newaxis, :]
# 		model = LogisticRegression(
# 			penalty='l1',
# 			solver='liblinear',
# 			C=1/alpha,
# 			tol=1e-3,
# 			warm_start=True,
# 			class_weight='balanced'
# 		)
# 		with warnings.catch_warnings():
# 			warnings.simplefilter('ignore')
# 			model.fit(X_weighted, y)
# 		coef = model.coef_.flatten() / weights
# 		coefficients[i, :] = (coef != 0).astype(int)
# 	return coefficients

# adaptive lasso regressor
def fit_adaptive_lasso_regressor(X, y, alphas, epsilon=1e-6):
	n_alphas = len(alphas)
	n_features = X.shape[1]
	coefficients = np.zeros((n_alphas, n_features))

	# get initial coefficients from ridge regression
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		init_ridge = Ridge()
		init_ridge.fit(X,y)
		init_coef = init_ridge.coef_

	# compute adaptive weights
	weights = 1 / (np.abs(init_coef) + epsilon)

	# solve weighted lasso for each alpha
	for i, alpha in enumerate(alphas):
		X_weighted = X / weights[np.newaxis, :]
		lasso = Lasso(alpha=alpha)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			lasso.fit(X_weighted, y)
		coef = lasso.coef_ / weights
		coefficients[i, :] = (coef != 0).astype(int)
	return coefficients

# gradient boosting classifier
def fit_gb_classifier(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gain')
	seed = np.random.randint(1e5)
	model = xgb.XGBClassifier(random_state=seed, **kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances

# gradient boosting regressor
def fit_gb_regressor(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gain')
	seed = np.random.randint(1e5)
	model = xgb.XGBRegressor(random_state=seed, **kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances

# l1-regularized logistic regression
def fit_l1_classifier(X, y, alphas, **kwargs):
	model = LogisticRegression(**kwargs)
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		model.set_params(C=1/alpha)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			model.fit(X, y)
			coefficients[i,:] = (model.coef_ != 0).astype(int)
	return coefficients

# l1-regularized linear regression (lasso)
def fit_l1_regressor(X, y, alphas, **kwargs):
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		_, coefs, _ = lasso_path(X, y, alphas=alphas, **kwargs)
		coefficients = (coefs.T != 0).astype(int)
	return coefficients

# minimax concave penalty (MCP)
def fit_mcp_regressor(X, y, alphas, **kwargs):
	gamma = kwargs.pop('gamma', 3.0)
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		model = MCPRegression(alpha=alpha, gamma=gamma, **kwargs)
		model.fit(X, y)
		coefficients[i, :] = (model.coef_ != 0).astype(int)
	return coefficients

# random forest classifier
def fit_rf_classifier(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gini')
	model = RandomForestClassifier(class_weight='balanced', **kwargs)
	model.fit(X, y)
	if importance_type == 'gini' or importance_type is None:
		feature_importances = model.feature_importances_
	elif importance_type == 'permutation':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
		perm_importance = permutation_importance(model, X_test, y_test, n_repeats=1)
		feature_importances = perm_importance.importances_mean
	else:
		raise ValueError("importance_type must be either 'gini' or 'permutation'")
	return feature_importances

# random forest regressor
def fit_rf_regressor(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'gini')
	model = RandomForestRegressor(**kwargs)
	model.fit(X, y)
	if importance_type == 'gini':
		feature_importances = model.feature_importances_
	elif importance_type == 'permutation':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
		perm_importance = permutation_importance(model, X_test, y_test, n_repeats=1)
		feature_importances = perm_importance.importances_mean
	elif importance_type == 'shadow':
		n, p = X.shape
		X_shadow = X.copy()
		for i in range(p):
			np.random.shuffle(X_shadow[:,i])
		X_combined = np.hstack((X, X_shadow))
		model.fit(X_combined, y)
		importances_combined = model.feature_importances_
		n_features = X.shape[1]
		original_importances = importances_combined[:n_features]
		shadow_importances = importances_combined[n_features:]
		feature_importances = original_importances - shadow_importances
	else:
		raise ValueError("importance_type must be either 'gini', 'permutation', or 'shadow'")
	return feature_importances

# smoothly clipped absolute deviation (SCAD)
def fit_scad_regressor(X, y, alphas, **kwargs):
	gamma = kwargs.pop('gamma', 3.7)
	coefficients = np.zeros((len(alphas), X.shape[1]))
	for i, alpha in enumerate(alphas):
		penalty = SCAD(alpha=alpha, gamma=gamma, **kwargs)
		model = GeneralizedLinearEstimator(penalty=penalty)
		model.fit(X, y)
		coefficients[i,:] = (model.coef_ != 0).astype(int)
	return coefficients


