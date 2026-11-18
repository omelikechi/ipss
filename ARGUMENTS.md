# `ipss` arguments

### Required arguments:
- `X`: Features (array of shape `(n,p)`), where `n` is the number of samples and `p` is the number of features.
- `y`: Response (array of shape `(n,)` or `(n, 1)`). `ipss` automatically detects if `y` is binary.

### Optional arguments:
- `selector`: Base algorithm to use (str; default `'gb'`).
	- **Nonparametric methods**
		- `'gb'`: Gradient boosting (XGBoost).
		- `'rf'`: Random forest (scikit-learn).
	- **Penalized regression methods**
		- `'adaptive_lasso'`: Adaptive lasso (scikit-learn)
		- `'l1'`: L1-regularized linear or logistic regression (scikit-learn).
		<!-- - `'mcp'`: Minimax concave penalty (skglm). -->
		<!-- - `'scad'`: Smoothly clipped absolute deviation penalty (skglm). -->
	- **Custom**
		- Users can provide their own feature importance function (see the [README](https://github.com/omelikechi/ipss#usage-with-custom-feature-importance-scores)). 
- `selector_args`: Arguments for the base algorithm (dict; default `None`). These are added to the selector's defaults.
- `preselect`: Preselect/filter features prior to subsampling (bool, int, or float; default `True`). An int keeps that many features, and a float in (0, 1] keeps that fraction.
- `preselector`: Method used to preselect features (default `None`, which resolves to the matching preselector for `'gb'`, `'l1'`, and `'adaptive_lasso'`, and to `'dcor'` for every other `selector`). Options:
	- `'gb'`: Gradient boosting (XGBoost) feature importances.
	- `'rf'`: Random forest (scikit-learn) feature importances.
	- `'l1'` or `'adaptive_lasso'`: Lasso (or l1 logistic regression) coefficients.
	- `'dcor'`: Distance correlation. Captures nonlinear relationships and is fast and deterministic.
	- Custom: a function `preselector(X, y, **preselector_args) -> array of length p` giving a feature importance score per feature; the top-scoring features (see `n_keep` below) are kept.
- `preselector_args`: Arguments for the preselection algorithm (dict; default `None`). Recognized keys:
	- `n_runs`: Number of times to refit the preselector, averaging feature importances across runs (int; default `3`; ignored by `'dcor'`, which is deterministic).
	- `n_keep`: Number of features to keep (int; defaults depend on `preselector`).
	- `expansion_factor`: For `'gb'` only, multiplies the number of features with nonzero importance to set `n_keep` (float; default `1.5`).
	- `engine`: For `'dcor'` only (str; default `'numpy'`). `'dcor'` uses the faster `dcor` package (`pip install ipss[dcor]`).
- `target_fp`: Target number of false positives to control (positive float; default `None`).
- `target_fdr`: Target false discovery rate (FDR) (float in (0, 1]; default `None`).
- `B`: Number of subsampling steps (int; default `100` if `selector` is `'gb'`, `50` otherwise).
- `n_alphas`: Number of values in the regularization or threshold grid (int; default `25` if `'l1'` else `100`).
- `ipss_function`: Function to apply to selection probabilities (str; default `'h2'` if `'l1'` else `'h3'`). Options:
	- `'h1'`: Linear function, ```h1(x) = 2x - 1 if x >= 0.5 else 0```.
	- `'h2'`: Quadratic function, ```h2(x) = (2x - 1)**2 if x >= 0.5 else 0```.
	- `'h3'`: Cubic function, ```h3(x) = (2x - 1)**3 if x >= 0.5 else 0```.
	- `'h4'`, `'h5'`, ...: Higher orders, ```hm(x) = (2x - 1)**m if x >= 0.5 else 0```. More aggressive than `'h3'`.
- `cutoff`: Maximum value of the theoretical integral bound `I(Lambda)` (positive float; default `0.05`).
- `delta`: Defines probability measure; see `Associated papers` (float; default `1`).
- `standardize_X`: Scale features to have mean 0, standard deviation 1 (bool; default `None`, which standardizes for `'l1'` and `'adaptive_lasso'` only).
- `center_y`: Center response to have mean 0 (bool; default `None`, which centers for `'l1'` and `'adaptive_lasso'` only). Binary responses are never centered.
- `subsample_size`: Size of each of the two disjoint subsamples (int; default `n // 2`).
- `stratify`: For a binary response, keep the class proportions of `y` in every subsample (bool; default `False`).
- `jitter`: Add a small amount of noise to categorical features (2 to 10 distinct values) so that tree selectors do not favor continuous features (default `None`, which jitters for `'gb'` and `'rf'`; also `True`, `False`, or a list of column indices).
- `n_jobs`: Number of jobs to run in parallel (int; default `1`).
- `random_state`: Seed for preselection, subsampling, and the `'gb'`, `'rf'`, and `'ufi'` selectors (int or `None`; default `None`, which uses NumPy's global random state). Each selector fit gets its own seed; `selector_args['random_state']` is ignored.

### General observations/recommendations:
- `selector = 'gb'` often best for capturing nonlinear relationships.
- `selector = 'l1'` or `'adaptive_lasso'` often best for capturing linear relationships.
- `preselector = 'dcor'` captures nonlinearities and is deterministic.
- We recommend computing q-values (or efp scores) and selecting features from them, leaving `target_fdr` and `target_fp` as `None`.
- In general, all other parameters should not be changed
	- `selector_args` include, e.g., decision tree parameters for tree-based models
	- Results are robust to `B` provided it is greater than `25`
	- `'h3'` is less conservative than `'h2'` which is less conservative than `'h1'`.
	- Preselection can significantly reduce computation time and increase power.
	- Results are robust to `cutoff` provided it is between `0.025` and `0.1`.
