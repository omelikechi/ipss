# Preselect features for IPSS to reduce dimensionality and computation time

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
import xgboost as xgb

# squared distance correlation between response and each feature
"""
"Measuring and testing dependence by correlation of distances" by Szekely et al (Annals of Statistics, 2007)
"""
def compute_dcor_scores(X, y):
	p = X.shape[1]
	b = np.abs(y[:, None] - y[None, :])
	B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
	dvar_y_sqr = np.mean(B * B)

	scores = np.zeros(p)
	for j in range(p):
		a = np.abs(X[:, j][:, None] - X[:, j][None, :])
		A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
		dcov_sqr = np.mean(A * B)
		dvar_x_sqr = np.mean(A * A)
		denom = dvar_x_sqr * dvar_y_sqr
		scores[j] = dcov_sqr / np.sqrt(denom) if denom > 0 else 0.0
	return scores

def preselection(X, y, preselector, preselector_args=None):
	n, p = X.shape

	if preselector_args is None:
		preselector_args = {}

	preselector_args_local = preselector_args.copy()
	n_runs = preselector_args_local.pop('n_runs', 3)
	n_keep = preselector_args_local.pop('n_keep', None)
	expansion_factor = preselector_args_local.pop('expansion_factor', 1.5)
	engine = preselector_args_local.pop('engine', 'numpy')

	preselect_indices = []

	if preselector in ['logistic_regression', 'lasso', 'adaptive_lasso_classifier', 'adaptive_lasso_regressor']:
		n_keep = n_keep or 200
		std_devs = np.std(X, axis=0)
		non_zero_std_indices = std_devs != 0
		X_filtered = X[:, non_zero_std_indices]
		correlations = np.array([np.abs(np.corrcoef(X_filtered[:, i], y)[0, 1]) for i in range(X_filtered.shape[1])])
		correlations = np.nan_to_num(correlations)

		alpha = max(np.sort(correlations)[::-1][min(p - 1, 2 * n_keep)], 1e-6)

		if preselector in ['logistic_regression', 'adaptive_lasso_classifier']:
			preselector_args_local = preselector_args_local or {'penalty':'l1', 'solver':'liblinear', 'tol':1e-3, 'class_weight':'balanced'}
			preselector_args_local.setdefault('C', 1 / alpha)
			model = LogisticRegression(**preselector_args_local)
		else:
			preselector_args_local.setdefault('alpha', alpha)
			model = Lasso(**preselector_args_local)

		feature_importances = np.zeros(p)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			for _ in range(n_runs):
				indices = np.random.choice(n, size=n, replace=True)
				X_sampled, y_sampled = X[indices], y[indices]
				model.fit(X_sampled, y_sampled)
				feature_importances += np.abs(model.coef_).ravel()
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	elif preselector in ['rf_classifier', 'rf_regressor', 'ufi_classifier', 'ufi_regressor']:
		n_keep = n_keep or 100
		preselector_args_local.setdefault('max_features', 0.1)
		preselector_args_local.setdefault('n_estimators', 25)
		model_class = RandomForestClassifier if 'classifier' in preselector else RandomForestRegressor
		model = model_class(**preselector_args_local)
		feature_importances = np.zeros(p)
		for _ in range(n_runs):
			model.set_params(random_state=np.random.randint(1e5))
			model.fit(X,y)
			feature_importances += model.feature_importances_
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	elif preselector in ['gb_classifier', 'gb_regressor']:
		preselector_args_local.setdefault('max_depth', 1)
		preselector_args_local.setdefault('colsample_bynode', 0.1)
		preselector_args_local.setdefault('n_estimators', 50)
		preselector_args_local.setdefault('importance_type', 'gain')
		model_class = xgb.XGBClassifier if preselector == 'gb_classifier' else xgb.XGBRegressor
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

	elif preselector == 'dcor':
		n_keep = n_keep or 200
		if engine == 'dcor':
			try:
				import dcor
			except ImportError as e:
				raise ImportError(
					"preselector='dcor' with engine='dcor' requires the dcor package. Install it with "
					"`pip install ipss[dcor]` or `pip install dcor`."
				) from e
			feature_importances = np.array([dcor.distance_correlation_sqr(X[:, j], y, method='auto') for j in range(p)])
		else:
			feature_importances = compute_dcor_scores(X, y)
		feature_importances = np.nan_to_num(feature_importances)
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	elif callable(preselector):
		n_keep = n_keep or 200
		feature_importances = np.zeros(p)
		for _ in range(n_runs):
			feature_importances += preselector(X, y, **preselector_args_local)
		preselect_indices = np.argsort(feature_importances)[::-1][:n_keep]

	else:
		raise ValueError(
			f"Unrecognized preselector: {preselector!r}. Must be one of 'gb', 'rf', 'ufi', 'l1', "
			"'adaptive_lasso', 'dcor', or a custom callable."
		)

	X_reduced = X[:, preselect_indices]

	return X_reduced, preselect_indices


