# Integrated path stability selection (IPSS)

Python implementation of integrated path stability selection (IPSS)

## Associated paper

arxiv:

## Installation
Install from PyPI:
```
pip install ipss
```
Clone from GitHub:
```
git clone git@github.com:omelikechi/ipss.git
```

## Usage

## Features

- **Flexible Regression Choices**: Supports least angle regression (LARS), Lasso regression, and logistic regression for binary outcomes.
- **Subsampling and Parallelization**: Implements subsampling to estimate feature stability across subsamples, with parallel processing for efficiency.
- **Custom Selection Functions**: Allows for the application of custom selection functions to estimated selection probabilities.
- **Stability Measure Incorporation**: Option to use a stability measure for feature selection, enhancing the robustness of selected features.
- **Sparse Matrix Support**: Outputs subsample tensor as a sparse matrix to handle high-dimensional data efficiently.

## Installation

To use IPSS, you will need Python installed on your system, along with the following dependencies:
- NumPy
- SciPy
- scikit-learn
- joblib

You can install these packages using `pip`
