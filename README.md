# Integrated path stability selection (IPSS)

Python implementation of integrated path stability selection (IPSS)

## Associated paper

arxiv:

## Installation
### Dependencies
- NumPy
- SciPy
- scikit-learn
- joblib

### Installing IPSS
To install from PyPI:
```
pip install ipss
```
To clone from GitHub:
```
git clone git@github.com:omelikechi/ipss.git
```

## Usage
Given an n-by-p matrix of features X (n = number of samples, p = number of features), an n-by-1 vector of responses y, and a user-specified target number of expected false positives, EFP: 
```python
from ipss import ipss

# Example: Load your data into X and y
# Define your expected number of false positives (EFP)
# Run IPSS:
result = ipss(X, y, EFP)

# Result analysis
print(result['selected_features'])  # features selected by IPSS
```
X and y are numpy arrays, and EFP is a positive scalar. Features are typically standardized, though this is left to the user to decide and implement beforehand. IPSS automatically detects whether y
is continuous (in which case lasso or LARS are used) or binary (in which case L1-regularized logistic regression is used).

### Results
`result` is a dictionary containing:
- `alphas`: Grid of regularization parameters (numpy array of size (n_alphas,)).
- `average_select`: Average number of variables selected at each regularization value (numpy array of size (n_alphas,)).
- `scores`: IPSS scores for each feature (numpy array of size (p,)).
- `selected_features`: Indices of the features selected by IPSS (list of integers whose length is the number of features selected by IPSS).
- `stability_paths`: Estimated selection probabilities for each feature across different regularization values (numpy array of size (n_alphas, p))
- `stop_index`: Index of the regularization value at which the IPSS threshold is passed (integer).
- `threshold`: The calculated threshold value tau = Integral / EFP (float).

### All arguments
`ipss` takes the following arguments:
- `X`: Feature matrix (numpy array of size (n,p)).
- `y`: Vector of responses, continuous or binary (numpy array with shape `(n,)` or `(n, 1)`).
- `EFP`: The target value for the expected number of false positives. It is a positive scalar with a default value of `1`.
- `cutoff`: A positive scalar C that, together with `EFP`, determines the IPSS threshold. The default value is `0.05`.
- `B`: Number of subsampling steps. It is an integer with a default value of `50`.
- `n_alphas`: Number of values in the grid of regularization parameters. It is an integer with a default value of `50`.
- `q_max`: Maximum number of features to be selected. If `None`, it defaults to `3p/4` if p < 200 and `p/2` otherwise. It can be an integer or `None`.
- `Z_sparse`: If `True`, the output tensor of subsamples, `Z`, is returned as sparse. The default is `False`.
- `lars`: If `True`, uses the least angle regression (LARS) for linear regression; otherwise, uses Lasso. The default is `False`.
- `selection_function`: Function to apply to the estimated selection probabilities. If equal to an integer, `m`, then the function is `h_m(x) = (2x - 1)**m` if `x >= 0.5` and `0` if `x < 0.5`. It can be an integer, a callable, or `None`. The default is `None`.
- `with_stability`: If `True`, uses a stability measure in the selection process. The default is `False`.
- `delta`: A scalar value that determines the scaling of the regularization interval. `delta = 1` corresponds to log scale, and `delta = 0` corresponds to linear scale. The default is `1`.
- `standardize_X`: If `True`, standardizes each feature in `X` by removing the mean and scaling to unit variance. The default is `True`.
- `center_y`: If `True`, centers `y` by subtracting the mean. This is only applicable for continuous response variables. The default is `True`.








