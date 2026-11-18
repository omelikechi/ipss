# Changelog

## 2.0.0

Results differ from 1.x for the same call. Rerunning a 1.x analysis will generally give larger efp scores and q-values.

### Fixed
- **E(FP) bound normalization.** When the cutoff shortened the grid, 1.x normalized the bound over the full grid but each feature's integral over the shortened one, so efp scores and q-values were too small and could exceed their targets several-fold. Both integrals now use the same interval and weights.
- **Choice of λ_min** follows Equation 3.3 of the IPSS paper.
- **Temporary plateaus** in the stability paths no longer stop the integration early.
- **Score scale.** Multiplying all importance scores by a constant no longer changes the results.
- **`selector_args`** are added to the selector's defaults instead of replacing them.
- **Reinsertion after preselection** keeps the retained efp scores.
- Several edge cases that crashed now return the null result.

### Changed
- **Default delta is 1** for every selector.
- **Default preselector for gb is gb** (was dcor). On correlated features at large p, dcor preselection exceeded the target FDR several-fold.
- **Jittering** (new, on by default for gb and rf) adds a small amount of noise to categorical features, so tree selectors do not favor continuous features.
- **Higher-order ipss functions** `'h4'`, `'h5'`, ... are available. Defaults are unchanged (`'h2'` for l1, `'h3'` otherwise).
- `preselect` also accepts a number or fraction of features to keep.
- `stratify=True` keeps the class proportions of a binary response in every subsample.
- Each tree fit gets its own seed; `selector_args['random_state']` is ignored with a warning.
- `preselector='ufi'` is no longer accepted (it was an ordinary random forest).
- Invalid inputs raise clear errors.
- Requires Python 3.8 or later.
