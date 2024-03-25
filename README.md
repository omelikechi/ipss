# Integrated path stability selection (IPSS)

Integrated path stability selection (IPSS) is a general method for improving feature selection algorithms that yields
more robust, accurate, and interpretable models. It does so by allowing users to control the expected number of 
falsely selected features, E(FP), while producing far more true positives than other versions of stability selection. 
This implementation of IPSS applied to L1-regularized linear and logistic regression is easy to use, requiring only 
the X (features) and y (response) data and specification of E(FP).

## Associated paper

arXiv:

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
pip install joblib numpy scikit-learn scipy
```
Additional dependencies required for examples:
```
pip install matplotlib seaborn
```

## Examples
Examples are available in the `examples` folder as both .py and .ipynb files. These include
- IPSS applied to data simulated from a multivariate normal. [Open in Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/simple/simple.ipynb)
- IPSS applied to prostate cancer data. [Open in Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/prostate/prostate.ipynb)
- IPSS applied to colon cancer data. [Open in Colab](https://colab.research.google.com/github/omelikechi/ipss/blob/main/examples/colon/colon.ipynb)

## Usage
Given an n-by-p numpy array of features, X (n = number of samples, p = number of features), an n-by-1 numpy array of responses, y, and a target number of expected false positives, EFP: 
```python
from ipss import ipss

# Load/generate X and y
# Specify EFP
# Run IPSS:
result = ipss(X, y, EFP)

# Print indices of features selected by IPSS
print(result['selected_features'])
```

### Results
`result = ipss(X, y, EFP)` is a dictionary containing:
- `alphas`: Grid of regularization parameters (array of shape `(n_alphas,)`).
- `average_select`: Average number of features selected at each regularization (array of shape `(n_alphas,)`).
- `scores`: IPSS score for each feature (array of shape `(p,)`).
- `selected_features`: Indices of features selected by IPSS (list of ints).
- `stability_paths`: Estimated selection probabilities at each regularization (array of shape `(n_alphas, p)`)
- `stop_index`: Index of regularization value at which IPSS threshold is passed (int).
- `threshold`: The calculated threshold value tau = Integral value / EFP (scalar).

### Full ist of arguments
`ipss` takes the following arguments (only `X` and `y` are required, and typically only `EFP` is specified):
- `X`: Features (array of shape `(n,p)`).
- `y`: Responses (array of shape `(n,)` or `(n, 1)`). IPSS automatically detects if `y` is continuous or binary.
- `EFP`: Target expected number of false positives (positive scalar; default is `1`).
- `cutoff`: Together with `EFP`, determines IPSS threshold (positive scalar; default is `0.05`).
- `B`: Number of subsampling steps (int; default is `50`).
- `n_alphas`: Number of values in regularization grid (int; default is `25`).
- `q_max`: Max number of features selected (int; default is `None`, in which case `q_max = p/2`).
- `Z_sparse`: If `True`, tensor of subsamples, `Z`, is sparse (default is `False`).
- `lars`: Implements least angle regression for linear regression if `True`, lasso otherwise (default is `False`).
- `selection_function`: Function to apply to the stability paths. If a positive int, `m`, function is `h_m(x) = (2x - 1)**m` if `x >= 0.5` and `0` if `x < 0.5` (int, callable, or `None`; default is `None`, in which case function is `h_2` if y is binary, or `h_3` if continuous).
- `with_stability`: If `True`, uses a stability measure in selection process (default is `False`).
- `delta`: Determines scaling of regularization interval (scalar; default is `1`).
- `standardize_X`: If `True`, standardizes all features (default is `True`).
- `center_y`: If `True`, centers `y` when it is continuous (default is `True`).








