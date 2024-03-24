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
Given an n-by-p matrix of features X (n = number of samples, p = number of features), an n-by-1 vector of responses y, and a target number of expected false positives, EFP: 
```python
from ipss import ipss

# Load data into X and y
# Define expected number of false positives (EFP)
# Run IPSS:
result = ipss(X, y, EFP)

# Result analysis
print(result['selected_features'])  # features selected by IPSS
```

### Results
`result` is a dictionary containing:
- `alphas`: Grid of regularization parameters (numpy array of size `(n_alphas,)`).
- `average_select`: Average number of variables selected at each regularization value (numpy array of size `(n_alphas,)`).
- `scores`: IPSS scores for each feature (numpy array of size `(p,)`).
- `selected_features`: Indices of the features selected by IPSS (list of ints).
- `stability_paths`: Estimated selection probabilities for each feature at each regularization value (numpy array of size `(n_alphas, p)`)
- `stop_index`: Index of the regularization value at which the IPSS threshold is passed (int).
- `threshold`: The calculated threshold value tau = Integral value / EFP (float).

### Full ist of arguments
`ipss` takes the following arguments:
- `X`: Feature matrix (numpy array of size `(n,p)`).
- `y`: Vector of responses (numpy array with shape `(n,)` or `(n, 1)`). IPSS automatically detects whether y is continuous (in which case it runs lasso or LARS) or binary (in which case it runs L1-regularized logistic regression).
- `EFP`: Target expected number of false positives (positive scalar; default is `1`).
- `cutoff`: Positive scalar that, together with `EFP`, determines the IPSS threshold (default is `0.05`).
- `B`: Number of subsampling steps (int; default is `50`).
- `n_alphas`: Number of values in regularization grid (int; default is `50`).
- `q_max`: Max number of features selected (int; default is `None`). If `None`, defaults to `3p/4` if p < 200 and `p/2` otherwise.
- `Z_sparse`: If `True`, output tensor of subsamples, `Z`, is sparse (default is `False`). Can help save space.
- `lars`: If `True`, uses least angle regression (LARS) for linear regression; otherwise, uses lasso (default is `False`).
- `selection_function`: Function to apply to the stability paths. If a positive int, `m`, function is `h_m(x) = (2x - 1)**m` if `x >= 0.5` and `0` if `x < 0.5` (int, callable, or `None`; default is `None`, in which case function is `h_2` if y is binary, or `h_3` if continuous).
- `with_stability`: If `True`, uses a stability measure in the selection process (default is `False`).
- `delta`: Scalar that determines the scaling of the regularization interval. `delta = 1` corresponds to log scale, and `delta = 0` corresponds to linear scale (default is `1`).
- `standardize_X`: If `True`, standardizes each feature in `X` by removing the mean and scaling to unit variance (default is `True`).
- `center_y`: If `True`, centers `y` by subtracting the mean. This is only applicable for continuous response variables (default is `True`).








