# Regularized linear models

import warnings

import numpy as np
from skglm.estimators import GeneralizedLinearEstimator, MCPRegression
from skglm.penalties import SCAD
from sklearn.linear_model import Lasso, lasso_path, LogisticRegression, Ridge, RidgeClassifier

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


