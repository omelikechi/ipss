# Preselect features for IPSS to reduce dimensionality and computation time

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
import xgboost as xgb

def preselection(X, y, selector, preselect, preselect_min, preselector_args=None, true_features=None):
	n, p = X.shape

	if p <= preselect_min:
		return X, np.arange(p), true_features

	if preselector_args is None:
		preselector_args = {}
	n_runs = preselector_args.get('n_runs', 3)

	if preselect <= 1:
		preselect = int(preselect * p)
	preselect = max(preselect, preselect_min)

	preselect_indices = []
	if selector == 'logistic_regression':
		# preselector_args = preselector_args or {'penalty': 'l1', 'solver': 'saga', 'tol': 1e-3, 'class_weight': 'balanced'}
		preselector_args = preselector_args or {'penalty':'l1', 'solver':'liblinear', 'max_iter':int(1e6), 'class_weight':'balanced'}
		
		std_devs = np.std(X, axis=0)
		non_zero_std_indices = std_devs != 0
		X_filtered = X[:, non_zero_std_indices]
		correlations = np.array([np.abs(np.corrcoef(X_filtered[:, i], y)[0, 1]) for i in range(X_filtered.shape[1])])
		correlations = np.nan_to_num(correlations)
		alpha = max(np.sort(correlations)[::-1][min(p - 1, 2 * preselect)], 1e-6)
		preselector_args.setdefault('C', 1 / alpha)
		model = LogisticRegression(**preselector_args)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			preselect_indices = fit_and_average(model, X, y, preselect, n_runs, linear_model=True)

	elif selector == 'lasso':
		preselector_args = preselector_args or {}
		std_devs = np.std(X, axis=0)
		non_zero_std_indices = std_devs != 0
		X_filtered = X[:, non_zero_std_indices]
		correlations = np.array([np.abs(np.corrcoef(X_filtered[:, i], y)[0, 1]) for i in range(X_filtered.shape[1])])
		correlations = np.nan_to_num(correlations)
		alpha = max(np.sort(correlations)[::-1][min(p - 1, 2 * preselect)], 1e-6)
		preselector_args.setdefault('alpha', alpha)
		model = Lasso(**preselector_args)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			preselect_indices = fit_and_average(model, X, y, preselect, n_runs, linear_model=True)

	elif selector in ['rf_classifier', 'rf_regressor']:
		preselector_args = preselector_args or {'max_features':'sqrt', 'n_estimators':100}
		model_class = RandomForestClassifier if selector == 'rf_classifier' else RandomForestRegressor
		model = model_class(**preselector_args)
		preselect_indices = fit_and_average(model, X, y, preselect, n_runs)

	elif selector in ['gb_classifier', 'gb_regressor']:
		preselector_args = preselector_args or {'max_depth':1, 'colsample_bynode':0.1, 'n_estimators':100, 'importance_type':'gain'}
		model_class = xgb.XGBClassifier if selector == 'gb_classifier' else xgb.XGBRegressor
		model = model_class(**preselector_args)
		preselect_indices = fit_and_average(model, X, y, preselect, n_runs)

	X_reduced = X[:, preselect_indices]

	new_true = []
	if true_features is not None:
		for i in range(len(preselect_indices)):
			if preselect_indices[i] in true_features:
				new_true.append(i)

	return X_reduced, preselect_indices, new_true

# helper function
def fit_and_average(model, X, y, n_keep, n_runs, linear_model=False):
	n, p = X.shape
	feature_importances = np.zeros(p)
	for i in range(n_runs):
		if linear_model:
			indices = np.random.choice(n, size=n, replace=True)
			X_sampled, y_sampled = X[indices], y[indices]
			model.fit(X_sampled, y_sampled)
			feature_importances += np.abs(model.coef_).ravel()
		else:
			model.set_params(random_state=np.random.randint(1e5))
			model.fit(X, y)
			feature_importances += model.feature_importances_
	average_importances = feature_importances / n_runs
	top_indices = np.argsort(average_importances)[::-1][:n_keep]
	return top_indices


