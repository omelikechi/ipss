# Integrated path stability selection (IPSS)

Integrated path stability selection (IPSS) is a general method for improving feature selection algorithms. Given an
n-by-p data matrix X (n = number of samples, p = number of features), and an n-dimensional response variable y, IPSS
applies a base selection algorithm to subsamples of the data to select features (columns of X) that are most related
to y. This package includes IPSS for gradient boosting (IPSSGB), L1-regularized linear models (IPSSL), and for random
forests (IPSSRF). The final outputs are **efp scores** and **q-values** for each feature in X.

- The **efp score** of feature j is the expected number of false positives selected when j is identified as significant. 
- The **q-value** of feature j is the false discovery rate (FDR) when feature j is identified as significant.

### Key features
- **Error control:** IPSS controls the number of false positives or the FDR, whichever is preferred.
- **Generality:** IPSSGB and IPSSRF are nonlinear, nonparametric methods. IPSSL is linear.
- **Speed:** IPSS is efficient. For example, IPSSGB runs in under 20 seconds on datasets with 500 samples and 5000 features.
- **Simplicity:** The only required inputs are `X` and `y`. Users can also specify the type of IPSS (IPSSGB, IPSSRF, or IPSSL), 
and either the target number of false positives or target FDR.

## Associated papers

**IPSSL:** [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877)
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
- `selector`: Base algorithm to use. Options are:
  - `'gb'`: Gradient boosting with XGBoost.
  - `'l1'`: L1-regularized linear (regression) or logistic (classification) regression.
  - `'rf'`: Random forest.  
  (Default is `'gb'`).
- `selector_args`: Arguments for the base algorithm (default is `None`).
- `target_fp`: Target number of false positives to control (positive scalar; default is `None`).
- `target_fdr`: Target false discovery rate (FDR) (positive scalar; default is `None`).
- `B`: Number of subsampling steps when computing selection probabilities (int; default is `50`).
- `n_alphas`: Number of values in the regularization or threshold grid (int; default is `25`).
- `ipss_function`: Function to apply to selection probabilities. Options are:
  - `'h1'`: Linear function, ```python h1(x) = 2*x - 1 if x >= 0.5 else 0```.
  - `'h2'`: Quadratic function, ```python h2(x) = (2*x - 1)**2 if x >= 0.5 else 0```.
  - `'h3'`: Cubic function, ```python h3(x) = (2*x - 1)**3 if x >= 0.5 else 0```.
  (Default is `'h3'`).
- `preselect`: Number (if int) or percentage (if float `0 < preselect <= 1`) of features to preselect. Set to `False` for no preselection (default is `0.05`).
- `preselect_min`: Minimum number of features to keep in the preselection step (int; default is `200`).
- `preselect_args`: Arguments for the preselection algorithm (default is `None`).
- `cutoff`: Maximum value of the theoretical integral bound `I(Lambda)` (positive scalar; default is `0.05`).
- `delta`: Defines the probability measure `mu_delta(dlambda) = z_delta^{-1}lambda^{-delta}dlambda` (scalar; default is `1`).
- `standardize_X`: Whether to standardize features to have mean 0 and standard deviation 1 (Boolean or `None`; default is `None`).
- `center_y`: Whether to center the response to have mean 0 (Boolean or `None`; default is `None`).
- `true_features`: List of true feature indices when known, for example, in simulation experiments (default is `None`).
- `n_jobs`: Number of jobs to run in parallel (int; default is `1`).








