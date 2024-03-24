# Integrated path stability selection (IPSS)

Python implementation of integrated path stability selection (IPSS)

## Associated paper

arxiv:

## Installation
Dependencies:
- NumPy
- SciPy
- scikit-learn
- joblib
  
Once these are installed, ipss can be installed from PyPI:
```
pip install ipss
```
or cloned from GitHub:
```
git clone git@github.com:omelikechi/ipss.git
```

## Usage
Given an n-by-p matrix of features X (n = number of samples, p = number of features), an n-by-1 vector of responses y, and a user-specified target number of expected false positives, EFP>0: 
```python
from ipss import ipss

% load and/or define X, y, and EFP

result = ipss(X, y, EFP)
```
X and y are numpy arrays, and EFP is a positive scalar. Features are typically standardized, though this is left to the user to decide and implement beforehand. IPSS automatically detects whether y
is continuous (in which case lasso or LARS are used) or binary (in which case L1-regularized logistic regression is used).

```result``` is a dictionary with the following keys:







