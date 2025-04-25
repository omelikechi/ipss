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

	preselector_args_local = preselector_args.copy()
	n_runs = preselector_args_local.pop('n_runs', 3)
	n_keep = preselector_args_local.pop('n_keep', None)
	expansion_factor = preselector_args_local.pop('expansion_factor', 1.5)

	preselect_indices = []

	if selector in ['lasso', 'logistic_regression']:
		n_keep = n_keep or 200
		std_devs = np.std(X, axis=0)
		non_zero_std_indices = std_devs != 0
		X_filtered = X[:, non_zero_std_indices]
		correlations = np.array([np.abs(np.corrcoef(X_filtered[:, i], y)[0, 1]) for i in range(X_filtered.shape[1])])
		correlations = np.nan_to_num(correlations)

		alpha = max(np.sort(correlations)[::-1][min(p - 1, 2 * n_keep)], 1e-6)

		if selector == 'lasso':
			preselector_args_local.setdefault('alpha', alpha)
			model = Lasso(**preselector_args_local)
		else:
			preselector_args_local = preselector_args_local or {'penalty':'l1', 'solver':'liblinear', 'tol':1e-3, 'class_weight':'balanced'}
			preselector_args_local.setdefault('C', 1 / alpha)
			model = LogisticRegression(**preselector_args_local)

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
		n_keep = n_keep or 100
		preselector_args_local.setdefault('max_features', 0.1)
		preselector_args_local.setdefault('n_estimators', 25)
		model_class = RandomForestClassifier if selector == 'rf_classifier' else RandomForestRegressor
		model = model_class(**preselector_args_local)
		feature_importances = np.zeros(p)
		for _ in range(n_runs):
			model.set_params(random_state=np.random.randint(1e5))
			model.fit(X,y)
			feature_importances += model.feature_importances_
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	elif selector in ['gb_classifier', 'gb_regressor']:
		preselector_args_local.setdefault('max_depth', 1)
		preselector_args_local.setdefault('colsample_bynode', 0.1)
		preselector_args_local.setdefault('n_estimators', 50)
		preselector_args_local.setdefault('importance_type', 'gain')
		model_class = xgb.XGBClassifier if selector == 'gb_classifier' else xgb.XGBRegressor
		model = model_class(**preselector_args_local)
		feature_importances = np.zeros(p)
		for i in range(n_runs):
			model.set_params(random_state=np.random.randint(1e5))
			model.fit(X, y)
			feature_importances += model.feature_importances_
		nonzero_mask = feature_importances > 0
		total_nonzero = np.sum(nonzero_mask)
		n_keep = min(p, max(100, int(expansion_factor * total_nonzero)))
		nonzero_indices = np.where(nonzero_mask)[0]
		# randomly sample zero-importance features
		if len(nonzero_indices) >= n_keep:
			preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]
		else:
			n_extra = n_keep - len(nonzero_indices)
			zero_indices = np.where(~nonzero_mask)[0]
			extra_indices = np.random.choice(zero_indices, size=n_extra, replace=False)
			preselect_indices = np.concatenate([nonzero_indices, extra_indices])

	else:
		n_keep = n_keep or 100
		feature_importances = np.zeros(p)
		for _ in range(n_runs):
			feature_importances += selector(X, y, **preselector_args_local)
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	X_reduced = X[:, preselect_indices]

	return X_reduced, preselect_indices


