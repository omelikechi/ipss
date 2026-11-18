# Integrated path stability selection (IPSS)

> Fast, flexible feature selection with false discovery control

## Papers

- **Integrated path stability selection**  
  [*Journal of the American Statistical Association*](https://doi.org/10.1080/01621459.2025.2525589) • [arXiv](https://arxiv.org/abs/2403.15877)

- **Nonparametric IPSS**  
  [*Bioinformatics*](https://doi.org/10.1093/bioinformatics/btaf299) • [arXiv](https://arxiv.org/abs/2410.02208)

## Installation
```
pip install ipss
```

## Usage
```python
from ipss import ipss

# load n-by-p feature matrix X and n-by-1 response vector y

# run ipss
ipss_output = ipss(X,y)

# select features based on target FDR
target_fdr = 0.1
q_values = ipss_output['q_values']
selected_features = [idx for idx, q_value in q_values.items() if q_value <= target_fdr]

print(f'Selected features (target FDR = {target_fdr}): {selected_features}')
```
### Outputs
`ipss_output = ipss(X,y)` is a dictionary containing:
- `efp_scores`: Dictionary whose keys are feature indices and values are their efp scores (dict of length `p`).
- `q_values`: Dictionary whose keys are feature indices and values are their q-values (dict of length `p`).
- `runtime`: Runtime of the algorithm in seconds (float).
- `selected_features`: Indices of selected features, if `target_fdr` or `target_fp` is given (list of ints).
- `stability_paths`: Estimated selection probabilities (array of shape `(k, p)`, with `k <= n_alphas`).

### Selecting features
Each feature (column of `X`) is assigned:
- a **q-value**: the minimum false discovery rate (FDR) at which the feature is selected
- an **efp score**: the minimum expected number of false positives (E(FP)) at which the feature is selected

To select features:
- **Control FDR** by choosing all features with `q_value ≤ target_fdr`  
  _Example: Selecting features with `q_value ≤ 0.1` controls the FDR at level 0.1_
- **Control E(FP)** by choosing all features with `efp_score ≤ target_fp`  
  _Example: Selecting features with `efp_score ≤ 2` controls the E(FP) at level 2_

## Main arguments
- `selector`: Base algorithm (default `'gb'`). `'gb'` (gradient boosting) and `'rf'` (random forest) capture nonlinear relationships; `'l1'` (lasso or l1 logistic regression) and `'adaptive_lasso'` capture linear ones. A custom function is also allowed (see below).
- `selector_args`: Arguments for the base algorithm (dict; default `None`).
- `n_jobs`: Number of jobs to run in parallel (int; default `1`).

See [ARGUMENTS.md](https://github.com/omelikechi/ipss/blob/main/ARGUMENTS.md) for the full list of arguments.

## Usage with custom feature importance scores
`selector` can also be a function that takes `X`, `y`, and the keyword arguments in `selector_args`, and returns one importance score per feature.
```python
import numpy as np
from sklearn.linear_model import Ridge
from ipss import ipss

# custom feature importance function based on ridge regression
def ridge_selector(X, y, alpha):
	return np.abs(Ridge(alpha=alpha).fit(X, y).coef_)

ipss_output = ipss(X, y, selector=ridge_selector, selector_args={'alpha': 1})
```

## Examples
The [examples](https://github.com/omelikechi/ipss/tree/main/examples) folder includes analyses of
- **Simulated data**: `simple_example.py` ([Open in Google Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/simple_example.ipynb)).
- **Cancer data**: `cancer.py` ([Open in Google Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/cancer.ipynb)).
