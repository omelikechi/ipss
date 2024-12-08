# Preselect features for IPSS to reduce dimensionality and computation time

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
import xgboost as xgb

def preselection(X, y, selector, preselector_args=None):
	n, p = X.shape

	if preselector_args is None:
		preselector_args = {}
	n_runs = preselector_args.pop('n_runs', 3)
	n_keep = preselector_args.pop('n_keep', None)
	prop_zero = preselector_args.pop('prop_zero', 1)

	preselect_indices = []

	if selector in ['lasso', 'logistic_regression']:

		if n_keep is None:
			# keep features based on correlation; linearly interpolate between (avg_max_cor, n_keep) = (0.75, 100) and (0.95, 25)
			avg_max_cor = average_max_correlation(X)
			n_keep = 100 if avg_max_cor <= 3/4 else int(-375 * avg_max_cor + 1525/4)

		std_devs = np.std(X, axis=0)
		non_zero_std_indices = std_devs != 0
		X_filtered = X[:, non_zero_std_indices]
		correlations = np.array([np.abs(np.corrcoef(X_filtered[:, i], y)[0, 1]) for i in range(X_filtered.shape[1])])
		correlations = np.nan_to_num(correlations)

		alpha = max(np.sort(correlations)[::-1][min(p - 1, 2 * n_keep)], 1e-6)

		if selector == 'lasso':
			preselector_args = preselector_args or {}
			preselector_args.setdefault('alpha', alpha)
			model = Lasso(**preselector_args)
		else:
			preselector_args = preselector_args or {'penalty':'l1', 'solver':'liblinear', 'tol':1e-3, 'class_weight':'balanced'}
			preselector_args.setdefault('C', 1 / alpha)
			model = LogisticRegression(**preselector_args)

		feature_importances = np.zeros(p)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			for _ in range(n_runs):
				indices = np.random.choice(n, size=n, replace=True)
				X_sampled, y_sampled = X[indices], y[indices]
				model.fit(X_sampled, y_sampled)
				feature_importances += np.abs(model.coef_).ravel()
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	elif selector in ['rf_classifier', 'rf_regressor']:

		if n_keep is None:
			n_keep = 100

		preselector_args = preselector_args or {'max_features':0.1, 'n_estimators':25}
		model_class = RandomForestClassifier if selector == 'rf_classifier' else RandomForestRegressor
		model = model_class(**preselector_args)

		feature_importances = np.zeros(p)
		for _ in range(n_runs):
			model.set_params(random_state=np.random.randint(1e5))
			model.fit(X,y)
			feature_importances += model.feature_importances_
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	elif selector in ['gb_classifier', 'gb_regressor']:
		preselector_args = preselector_args or {'max_depth':1, 'colsample_bynode':0.1, 'n_estimators':25, 'importance_type':'gain'}
		model_class = xgb.XGBClassifier if selector == 'gb_classifier' else xgb.XGBRegressor
		model = model_class(**preselector_args)

		nonzero_counts = np.zeros(p, dtype=int)
		for _ in range(n_runs):
			model.set_params(random_state=np.random.randint(1e5))
			model.fit(X,y)
			importances = model.feature_importances_
			nonzero_counts += (importances > 0).astype(int)
		preselect_indices = np.where(nonzero_counts != 0)[0]

		if prop_zero > 0:
			excluded_features = np.setdiff1d(np.arange(p), preselect_indices)
			n_included = len(preselect_indices)
			n_null = min(len(excluded_features), int(prop_zero * n_included))
			reintroduced_features = np.random.choice(excluded_features, size=n_null, replace=False)
			preselect_indices = np.concatenate((preselect_indices, reintroduced_features))

	X_reduced = X[:, preselect_indices]

	return X_reduced, preselect_indices

def average_max_correlation(X):
	corr_matrix = np.corrcoef(X, rowvar=False)
	abs_corr_matrix = np.abs(corr_matrix)
	np.fill_diagonal(abs_corr_matrix, 0)
	max_correlations = np.max(abs_corr_matrix, axis=1)
	avg_max_correlation = np.mean(max_correlations)
	
	return avg_max_correlation


