# Base selectors package
"""
Expose all base estimator functions so they can be imported directly from ipss.base_selectors.
"""

from .linear import (
	fit_l1_regressor,
	fit_l1_classifier,
	fit_adaptive_lasso_classifier,
	fit_adaptive_lasso_regressor,
	fit_mcp_regressor,
	fit_scad_regressor,
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
	"fit_l1_regressor",
	"fit_l1_classifier",
	"fit_adaptive_lasso_classifier",
	"fit_adaptive_lasso_regressor",
	"fit_mcp_regressor",
	"fit_scad_regressor",
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
