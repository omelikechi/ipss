# Integrated path stability selection (IPSS)

Integrated path stability selection (IPSS) is a general method for improving feature selection algorithms. Given an
n-by-p data matrix X (n = number of samples, p = number of features), and an n-dimensional response variable y, IPSS
applies a base selection algorithm to subsamples of the data to select features (columns of X) that are most related
to y. This package includes IPSS for gradient boosting (IPSSGB), L1-regularized linear models (IPSSL), and for random
forests (IPSSRF). The final outputs are **efp scores** and **q-values** for each feature in X.

- The **efp score** of feature j is the expected number of false positives selected when j is selected. 
- The **q-value** of feature j is the false discovery rate (FDR) when feature j is selected.

### Key features
- **Error control:** IPSS controls the number of false positives and the FDR.
- **Generality:** IPSSGB and IPSSRF are nonlinear, nonparametric methods. IPSSL is linear.
- **Speed:** IPSS is efficient. For example, IPSSGB runs in <20 seconds when `n = 500` and `p = 5000`.
- **Simplicity:** The only required inputs are `X` and `y`. Users can also specify the base method (IPSSGB, IPSSRF, or IPSSL), 
and the target number of false positives or the target FDR.

## Associated papers

**IPSSL:** [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877) <br>
**IPSSGB and IPSSRF:** 

## Installation
To install from PyPI:
```
pip install ipss
```
To clone from GitHub:
```
git clone git@github.com:omelikechi/ipss.git
```
Or clone from GitHub using HTTPS:
```
git clone https://github.com/omelikechi/ipss.git
```
### Dependencies
For `ipss`:
```
pip install joblib numpy scikit-learn xgboost
```
Additional dependencies required for examples:
```
pip install matplotlib
```

## Examples
Examples are available in the [examples](https://github.com/omelikechi/ipss/tree/main/examples) folder as both .py and .ipynb files. These include
<!-- - IPSS applied to data simulated from a multivariate normal. [Open in Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/simple/simple.ipynb)
- IPSS applied to prostate cancer data. [Open in Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/prostate/prostate.ipynb)
- IPSS applied to colon cancer data. [Open in Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/colon/colon.ipynb) -->

## Usage
Given an n-by-p numpy array of features, X (n = number of samples, p = number of features), and an n-by-1 numpy array of responses:
```python
from ipss import ipss

# load data X and y

# run ipss:
ipss_output = ipss(X, y)

# select features based on target number of false positives
target_fp = 1
efp_scores = ipss_output['efp_scores']
selected_features = []
for feature_index, efp_score in efp_scores:
	if efp_score <= target_fp:
		selected_features.append(feature_index)
print(f'Selected features (target E(FP) = {target_fp}): {selected_features}')

# select features based on target FDR
target_fdr = 0.1
q_values = ipss_output['q_values']
selected_features = []
for feature_index, q_value in q_values:
	if q_value <= target_fdr:
		selected_features.append(feature_index)
print(f'Selected features (target FDR = {target_fdr}): {selected_features}')
```

### Results
`ipss_output = ipss(X, y)` is a dictionary containing:
- `efp_scores`: List of tuples `(feature_index, efp_score)` with features ordered by their efp scores from smallest to largest (list of length `p`).
- `q_values`: List of tuples `(feature_index, q_value)` with features ordered by their q-values from smallest to largest (list of length `p`).
- `runtime`: The runtime of algorithm in seconds.
- `selected_features`: List of indices of features selected by IPSS; empty list if `target_fp` and `target_fdr` are not specified (list of ints).
- `stability_paths`: Estimated selection probabilities at each parameter value (array of shape `(n_alphas, p)`)

### Full list of arguments
`ipss` takes the following arguments (only `X` and `y` are required):
- `X`: Features (array of shape `(n, p)`), where `n` is the number of samples and `p` is the number of features.
- `y`: Response (array of shape `(n,)` or `(n, 1)`). IPSS automatically detects if `y` is continuous or binary.
#### Optional Arguments:
- `selector`: Base algorithm to use (default `'gb'`). Options are:
  - `'gb'`: Gradient boosting with XGBoost.
  - `'l1'`: L1-regularized linear or logistic regression.
  - `'rf'`: Random forest. 
- `selector_args`: Arguments for the base algorithm (default `None`).
- `target_fp`: Target number of false positives to control (positive scalar; default `None`).
- `target_fdr`: Target false discovery rate (FDR) (positive scalar; default `None`).
- `B`: Number of subsampling steps (int; default `100` for IPSSGB, `50` otherwise).
- `n_alphas`: Number of values in the regularization or threshold grid (int; default `100`).
- `ipss_function`: Function to apply to selection probabilities (default `'h3'`). Options are:
  - `'h1'`: Linear function, ```h1(x) = 2x - 1 if x >= 0.5 else 0```.
  - `'h2'`: Quadratic function, ```h2(x) = (2x - 1)**2 if x >= 0.5 else 0```.
  - `'h3'`: Cubic function, ```h3(x) = (2x - 1)**3 if x >= 0.5 else 0```.
- `preselect`: Number (if int) or percentage (if float `0 < preselect <= 1`) of features to preselect. Set to `False` for no preselection (default `0.05`).
- `preselect_min`: Minimum number of features to keep in the preselection step (int; default `200`).
- `preselect_args`: Arguments for the preselection algorithm (default `None`).
- `cutoff`: Maximum value of the theoretical integral bound `I(Lambda)` (positive scalar; default `0.05`).
- `delta`: Defines probability measure; see `Associated papers` (scalar; default `1`).
- `standardize_X`: Scale features to have mean 0, standard deviation 1 (Boolean or `None`; default `None`).
- `center_y`: Center response to have mean 0 (Boolean or `None`; default `None`).
- `true_features`: List of true feature indices when known, e.g., in simulations (default `None`).
- `n_jobs`: Number of jobs to run in parallel (int; default `1`).

#### General observations/recommendations:
- IPSSGB is usually best for capturing nonlinear relationships between features and response
- IPSSL is usually best for capturing linear relationships between features and response
- `target_fp` or `target_fdr` (at most one is specified) are problem specific/left to the user
- In general, all other parameters should not changed
  - `selector_args` include, e.g., decision tree parameters for tree-based models
  - Results are robust to `B` provided it is bigger than `25`
  - Results are robust to `n_alphas` provided it is bigger than `50`
  - `'h3'` yields the most true positives. `'h2'` is more conservative, and `'h1'` even more so.
  - Preselection can significantly reduce computation time. Results are robust otherwise.
  - Results are robust to `cutoff` provided it is between `0.025` and `0.1`.
  - Results are robust to `delta` provided it is between `0` and `1.5`.
  - Standardization is automatically applied for IPSSL. IPSSGB and IPSSRF are unaffected by this.
  - Centering `y` is automatically applied for IPSSL. IPSSGB and IPSSRF are unaffected by this.










