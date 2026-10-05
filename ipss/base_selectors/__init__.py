# Base selectors package
"""
Expose all base estimator functions so they can be imported directly from ipss.base_selectors.
Update (version 1.1.7): SCAD and MCP temporarily disabled due to skglm dependency issues
"""

from .linear import (
	adaptive_weights,
	fit_l1_regressor,
	fit_l1_classifier,
	fit_adaptive_lasso_classifier,
	fit_adaptive_lasso_regressor,
)

from .rf import (
	fit_rf_classifier,
	fit_rf_regressor,
)

from .gb import (
	fit_gb_classifier,
	fit_gb_regressor,
)

from .ufi import (
	fit_ufi_classifier,
	fit_ufi_regressor,
)

__all__ = [
	# linear
	"adaptive_weights",
	"fit_l1_regressor",
	"fit_l1_classifier",
	"fit_adaptive_lasso_classifier",
	"fit_adaptive_lasso_regressor",
	# random forest
	"fit_rf_classifier",
	"fit_rf_regressor",
	# gradient boosting
	"fit_gb_classifier",
	"fit_gb_regressor",
	# unbiased feature importance
	"fit_ufi_classifier",
	"fit_ufi_regressor",
]
