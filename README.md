# Integrated path stability selection (IPSS)

Integrated path stability selection (IPSS) is a general method for improving feature selection algorithms. Given an
n-by-p data matrix X (n = number of samples, p = number of features), and an n-dimensional response variable y, IPSS
applies a base selection algorithm to subsamples of the data to select features (columns of X) that are most related
to y. This package includes IPSS for gradient boosting (IPSSGB), random forests (IPSSRF), and L1-regularized linear 
models (IPSSL). The final outputs are **efp scores** and **q-values** for each feature in X.

- The **efp score** of feature j is the expected number of false positives, E(FP), selected when j is selected.
  - So to control the E(FP) at `target_fp`, select the features with efp scores at most `target_fp`. 
- The **q-value** of feature j is the smallest false discovery rate (FDR) when feature j is selected.
  - So to control the FDR at `target_fdr`, select the features with q-values at most `target_fdr`. 

### Key attributes
- **Error control:** IPSS controls the number of false positives and the FDR.
- **Generality:** IPSSGB and IPSSRF are nonlinear, nonparametric methods. IPSSL is linear.
- **Speed:** IPSS is efficient. For example, IPSSGB runs in <20 seconds when `n = 500` and `p = 5000`.
- **Simplicity:** The only required inputs are `X` and `y`. Users can also specify the base method (IPSSGB, IPSSRF, or IPSSL), 
and the target number of false positives or the target FDR.

## Associated papers

**IPSSL:** [https://arxiv.org/abs/2403.15877](https://arxiv.org/abs/2403.15877) <br>
**IPSSGB and IPSSRF:** [https://arxiv.org/abs/2410.02208v1](https://arxiv.org/abs/2410.02208v1)

## Installation
Install from PyPI:
```
pip install ipss
```

## Tests
**Basic test** (`tests/test.py`)
- Run the test:
```
python3 -m tests.test
```
- Expected output: "All tests passed."

## Usage
```python
from ipss import ipss

# load n-by-p feature matrix X and n-by-1 response vector y

# run ipss:
ipss_output = ipss(X,y)

# select features based on target FDR
target_fdr = 0.1
q_values = ipss_output['q_values']
selected_features = [idx for idx, q_value in q_values.items() if q_value <= target_fdr]
print(f'Selected features (target FDR = {target_fdr}): {selected_features}')
```
### Results
`ipss_output = ipss(X, y)` is a dictionary containing:
- `efp_scores`: Dictionary whose keys are feature indices and values are their efp scores (dict of length `p`).
- `q_values`: Dictionary whose keys are feature indices and values are their q-values (dict of length `p`).
- `runtime`: Runtime of the algorithm in seconds (float).
- `selected_features`: List of indices of features selected by IPSS; empty list if `target_fp` and `target_fdr` are not specified (list of ints).
- `stability_paths`: Estimated selection probabilities at each parameter value (array of shape `(n_alphas, p)`)

## Examples
The [examples](https://github.com/omelikechi/ipss/tree/main/examples) folder includes
- **A simple simulation**: `simple_example.py` ([Open in Google Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/simple_example.ipynb)).
- **Analyze cancer data**: `cancer.py` ([Open in Google Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/cancer.ipynb)).

## Full list of `ipss` arguments

### Required arguments:
- `X`: Features (array of shape `(n, p)`), where `n` is the number of samples and `p` is the number of features.
- `y`: Response (array of shape `(n,)` or `(n, 1)`). `ipss` automatically detects if `y` is continuous or binary.

### Optional arguments:
- `selector`: Base algorithm to use (str; default `'gb'`). Options:
  - `'gb'`: Gradient boosting (uses XGBoost).
  - `'l1'`: L1-regularized linear or logistic regression (uses sci-kit learn).
  - `'rf'`: Random forest (uses sci-kit learn). 
- `selector_args`: Arguments for the base algorithm (dict; default `None`).
- `preselect`: Preselect/filter features prior to subsampling (bool; default `True`).
- `preselect_args`: Arguments for preselection algorithm (dict; default `None`).
- `target_fp`: Target number of false positives to control (positive float; default `None`).
- `target_fdr`: Target false discovery rate (FDR) (positive float; default `None`).
- `B`: Number of subsampling steps (int; default `100` for IPSSGB, `50` otherwise).
- `n_alphas`: Number of values in the regularization or threshold grid (int; default `25` if `'l1'` else `100`).
- `ipss_function`: Function to apply to selection probabilities (str; default `'h2'` if `'l1'` else `'h3'`). Options:
  - `'h1'`: Linear function, ```h1(x) = 2x - 1 if x >= 0.5 else 0```.
  - `'h2'`: Quadratic function, ```h2(x) = (2x - 1)**2 if x >= 0.5 else 0```.
  - `'h3'`: Cubic function, ```h3(x) = (2x - 1)**3 if x >= 0.5 else 0```.
- `cutoff`: Maximum value of the theoretical integral bound `I(Lambda)` (positive float; default `0.05`).
- `delta`: Defines probability measure; see `Associated papers` (float; defaults depend on `selector`).
- `standardize_X`: Scale features to have mean 0, standard deviation 1 (bool; default `None`).
- `center_y`: Center response to have mean 0 (bool; default `None`).
- `n_jobs`: Number of jobs to run in parallel (int; default `1`).

### General observations/recommendations:
- IPSSGB is usually best for capturing nonlinear relationships between features and response
- IPSSL is usually best for capturing linear relationships between features and response
- For FDR control, it is generally better to run `ipss` to compute q-values first, then select features based on the desired FDR threshold, as in the Usage example.
- In general, all other parameters should not changed
  - `selector_args` include, e.g., decision tree parameters for tree-based models
  - Results are robust to `B` provided it is greater than `25`
  - `'h3'` is less conservative than `'h2'` which is less conservative `'h1'`.
  - Preselection can significantly reduce computation time.
  - Results are robust to `cutoff` provided it is between `0.025` and `0.1`.
  - Results are robust to `delta` provided it is between `0` and `1`.
  - Standardization is automatically applied for IPSSL. IPSSGB and IPSSRF are unaffected by this.
  - Centering `y` is automatically applied for IPSSL. IPSSGB and IPSSRF are unaffected by this.










