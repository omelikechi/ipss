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
Given an n-by-p matrix of features, X (n = number of samples, p = number of features), an n-by-1 vector of responses, y, and a target number of expected false positives, EFP: 
```python
from ipss import ipss

# Load data X and y
# Specify expected number of false positives (EFP)
# Run IPSS:
result = ipss(X, y, EFP)

# Result analysis
print(result['selected_features'])  # features selected by IPSS
```

### Results
`result` is a dictionary containing:
- `alphas`: Grid of regularization parameters (array of shape `(n_alphas,)`).
- `average_select`: Average number of variables selected at each regularization value (array of shape `(n_alphas,)`).
- `scores`: IPSS scores for each feature (array of shape `(p,)`).
- `selected_features`: Indices of the features selected by IPSS (list of ints).
- `stability_paths`: Estimated selection probabilities at each regularization value (array of shape `(n_alphas, p)`)
- `stop_index`: Index of the regularization value at which the IPSS threshold is passed (int).
- `threshold`: The calculated threshold value tau = Integral value / EFP (scalar).

### Full ist of arguments
`ipss` takes the following arguments (though it only requires `X` and `y`, and typically only `EFP` is altered):
- `X`: Feature matrix (array of shape `(n,p)`).
- `y`: Vector of responses (array of shape `(n,)` or `(n, 1)`). IPSS automatically detects whether y is continuous (in which case it runs lasso or LARS) or binary (in which case it runs L1-regularized logistic regression).
- `EFP`: Target expected number of false positives (positive scalar; default is `1`).
- `cutoff`: Together with `EFP`, determines IPSS threshold (positive scalar; default is `0.05`).
- `B`: Number of subsampling steps (int; default is `50`).
- `n_alphas`: Number of values in regularization grid (int; default is `50`).
- `q_max`: Max number of features selected (int; default is `None`). If `None`, defaults to `3p/4` if p < 200 and `p/2` otherwise.
- `Z_sparse`: If `True`, output tensor of subsamples, `Z`, is sparse (default is `False`). Can help save space.
- `lars`: Implements least angle regression (LARS) for linear regression if `True`, lasso otherwise (default is `False`).
- `selection_function`: Function to apply to the stability paths. If a positive int, `m`, function is `h_m(x) = (2x - 1)**m` if `x >= 0.5` and `0` if `x < 0.5` (int, callable, or `None`; default is `None`, in which case function is `h_2` if y is binary, or `h_3` if continuous).
- `with_stability`: If `True`, uses a stability measure in the selection process (default is `False`).
- `delta`: Determines scaling of regularization interval (float; default is `1`).
- `standardize_X`: If `True`, standardizes each feature in `X` (default is `True`).
- `center_y`: If `True`, centers `y` if it is continuous (default is `True`).








