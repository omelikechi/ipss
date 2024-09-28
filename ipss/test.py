# Simple test for IPSS

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_regression, SelectFdr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from main import ipss

# set random seed for reproducibility
np.random.seed(32123)

target_fdr = 0.1

# simulation parameters
n = 100
p = 200
n_true = 10
snr = 2

# truth
beta = np.zeros(p)
truth_and_correlates = np.random.choice(p, size=2 * n_true, replace=False)
truth = truth_and_correlates[:n_true]
correlates = truth_and_correlates[n_true:]
beta[truth] = np.random.normal(0, 1, size=(n_true))

# features
X = np.random.normal(0, 1, size=(n,p))
X[:,correlates] = X[:,truth] + np.random.normal(0, 0.5, size=(n,n_true))
X = StandardScaler().fit_transform(X)

for i in range(n_true):
	print(np.round(X[:,correlates[i]].T @ X[:,truth[i]] / n, 2))

# signal
signal = X @ beta

# noise
noise = np.sqrt(np.var(signal) / snr)

# response
y = signal + np.random.normal(0, noise, size=n)

# Step 1: Perform univariate regression for each feature
# Use f_regression to compute F-statistic and p-values for each feature
f_values, p_values = f_regression(X, y)

# Step 2: Apply Benjamini-Hochberg (BH) correction for controlling FDR
_, pvals_corrected, _, _ = multipletests(p_values, alpha=target_fdr, method='fdr_bh')

# Step 3: Select features where corrected p-values are below the FDR threshold
selected_features = np.where(pvals_corrected < target_fdr)[0]

tp, cor, uncor = 0, 0, 0
for feature in selected_features:
	if feature in truth:
		tp += 1
	elif feature in correlates:
		cor += 1
	else:
		uncor += 1
print(f'----- FDR -----')
print(f'tp, cor, uncor = {tp}, {cor}, {uncor}')

# run ipss
result = ipss(X, y, selector='l1', preselect=0.05, target_fdr=target_fdr)

efp_scores = result['efp_scores']
q_values = result['q_values']

selected_features = result['selected_features']

tp, cor, uncor = 0, 0, 0
for feature in selected_features:
	if feature in truth:
		tp += 1
	elif feature in correlates:
		cor += 1
	else:
		uncor += 1
print(f'----- ipssl -----')
print(f'tp, cor, uncor = {tp}, {cor}, {uncor}')

# # run ipss
# result = ipss(X, y, selector='gb', preselect=0.05, target_fdr=0.05)

# efp_scores = result['efp_scores']
# q_values = result['q_values']

# selected_features = result['selected_features']

# tp, cor, uncor = 0, 0, 0
# for feature in selected_features:
# 	if feature in truth:
# 		tp += 1
# 	elif feature in correlates:
# 		cor += 1
# 	else:
# 		uncor += 1
# print(f'----- ipssgb -----')
# print(f'tp, cor, uncor = {tp}, {cor}, {uncor}')

stability_paths = result['stability_paths']
n_alphas, p = stability_paths.shape

colors = ['dodgerblue' if i in truth else 'orange' if i in correlates else 'gray' for i in range(p)]

for j in range(p):
	plt.plot(np.arange(n_alphas), stability_paths[:,j], color=colors[j])
plt.tight_layout()
plt.show()



