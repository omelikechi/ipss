# Feature importance scores from gradient boosted trees (XGBoost)

import numpy as np
import xgboost as xgb

# gradient boosting classifier
def fit_gb_classifier(X, y, **kwargs):
	model = xgb.XGBClassifier(**kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances

# gradient boosting regressor
def fit_gb_regressor(X, y, **kwargs):
	model = xgb.XGBRegressor(**kwargs)
	model.fit(X,y)
	feature_importances = model.feature_importances_
	return feature_importances