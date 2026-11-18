# Feature importance scores from gradient boosted trees (XGBoost)

import numpy as np
import xgboost as xgb

# gradient boosting classifier
def fit_gb_classifier(X, y, **kwargs):
	if kwargs.get('random_state') is None:  # XGBoost would otherwise use seed 0 for every fit
		kwargs['random_state'] = np.random.randint(np.iinfo(np.int32).max)
	model = xgb.XGBClassifier(**kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances

# gradient boosting regressor
def fit_gb_regressor(X, y, **kwargs):
	if kwargs.get('random_state') is None:  # XGBoost would otherwise use seed 0 for every fit
		kwargs['random_state'] = np.random.randint(np.iinfo(np.int32).max)
	model = xgb.XGBRegressor(**kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances
