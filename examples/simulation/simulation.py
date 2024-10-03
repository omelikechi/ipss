# Run IPSS on simulated data

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from ipss.main import ipss
from simulation_helpers import generate_features, generate_response

# set random seed
np.random.seed(302)

#--------------------------------
# Generate data
#--------------------------------
# simulation parameters
n = 500 # number of samples
p = 500 # number of features
n_true = 20 # number of true features
snr = 2 # signal-to-noise ratio
nonlinear = True # linear or nonlinear relationship between features and response
response_type = 'continuous' # 'binary' for classification or 'continuous' for regression

# generate features
"""
Options for feature_type are:
	- 'standard_normal' for features from standard normal distribution
	- 'toeplitz' for features from Toeplitz model with correlation parameter 0 <= rho <= 1 (larger rho = more correlation)
	- 'ovarian_rnaseq' for features from real RNA-seq data from ovarian cancer patients (The Cancer Genome Atlas)
See also the generate_features function in simulation_helpers.py
"""
X = generate_features(n, p, feature_type='ovarian_rnaseq', rho=0.5)

# generate response and true features
y, true_features = generate_response(X, n_true, snr, nonlinear=nonlinear, response_type=response_type)

#--------------------------------
# Run IPSS
#--------------------------------
ipss_output = ipss(X, y, selector='gb')

#--------------------------------
# Analyze results
#--------------------------------
"""
Note: This is just one simulation; actual E(FP) and FDR performance is best seen by averaging over repeated simulations
"""
plot_target_fdr = True
plot_target_efp = True
plot_stability_paths = True

if plot_target_fdr:
	target_fdrs = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
	q_values = ipss_output['q_values']
	tprs, fdrs = np.zeros_like(target_fdrs), np.zeros_like(target_fdrs)
	for i, target_fdr in enumerate(target_fdrs):
		selected_features = [feature for (feature, q_value) in q_values if q_value <= target_fdr]
		tp, fp = 0, 0
		for feature in selected_features:
			if feature in true_features:
				tp += 1
			else:
				fp += 1
		tprs[i] = tp / n_true
		fdrs[i] = 0 if tp + fp == 0 else fp / (tp + fp)

	fig, ax = plt.subplots(1, 2, figsize=(14,6))
	ax[0].plot(target_fdrs, tprs)
	ax[1].plot(target_fdrs, fdrs)

	ax[0].set_xlabel(f'Target FDR')
	ax[1].set_xlabel(f'Target FDR')

	ax[0].set_ylabel(f'True positive rate')
	ax[1].set_ylabel(f'False positive rate')

	plt.tight_layout()
	plt.show()

if plot_target_efp:
	target_efps = [1/8, 1/4, 1/2, 1, 2, 3, 4, 5]
	efp_scores = ipss_output['efp_scores']
	tps, fps = np.zeros_like(target_efps), np.zeros_like(target_efps)
	for i, target_efp in enumerate(target_efps):
		selected_features = [feature for (feature, efp_score) in efp_scores if efp_score <= target_efp]
		for feature in selected_features:
			if feature in true_features:
				tps[i] += 1
			else:
				fps[i] += 1
	fig, ax = plt.subplots(1, 2, figsize=(14,6))
	ax[0].plot(target_efps, tps)
	ax[1].plot(target_efps, fps)

	ax[0].set_xlabel(f'Target E(FP)')
	ax[1].set_xlabel(f'Target E(FP)')

	ax[0].set_ylabel(f'True positives')
	ax[1].set_ylabel(f'False positives')

	plt.tight_layout()
	plt.show()

if plot_stability_paths:
	stability_paths = ipss_output['stability_paths']
	n_alphas, p = stability_paths.shape

	# blue paths for true features, gray for false features
	color = ['dodgerblue' if i in true_features else 'gray' for i in range(p)]

	for j in range(p):
		plt.plot(np.arange(n_alphas), stability_paths[:,j], color=color[j])
	plt.tight_layout()
	plt.show()

