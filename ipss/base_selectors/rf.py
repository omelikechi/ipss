# Feature importance scores from random forests (scikit-learn)

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# random forest classifier
def fit_rf_classifier(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'mdi')
	model = RandomForestClassifier(class_weight='balanced', **kwargs)
	model.fit(X, y)
	if importance_type == 'mdi':
		feature_importances = model.feature_importances_
	elif importance_type == 'permutation':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
		perm_importance = permutation_importance(model, X_test, y_test, n_repeats=1)
		feature_importances = perm_importance.importances_mean
	else:
		raise ValueError("importance_type must be either 'mdi' or 'permutation'")
	return feature_importances

# random forest regressor
def fit_rf_regressor(X, y, **kwargs):
	importance_type = kwargs.pop('importance_type', 'mdi')
	model = RandomForestRegressor(**kwargs)
	model.fit(X, y)
	if importance_type == 'mdi':
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
		raise ValueError("importance_type must be either 'mdi', 'permutation', or 'shadow'")
	return feature_importances