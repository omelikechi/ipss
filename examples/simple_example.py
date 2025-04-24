# Simple example using IPSS

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from ipss import ipss

# set random seed
np.random.seed(302)

#--------------------------------
# Generate data
#--------------------------------
n = 250 # number of samples
p = 500 # number of features
n_true = 20 # number of true features
snr = 2 # signal-to-noise ratio

# generate standard normal data
X = np.random.normal(0, 1, size=(n,p))

# randomly select true features
true_features = np.random.choice(p, size=n_true, replace=False)

# generate and center response variable y
beta = np.zeros(p)
beta[true_features] = np.random.normal(0, 1, size=(n_true))
signal = X @ beta
noise = np.sqrt(np.var(signal) / snr)
y = signal + np.random.normal(0, noise, size=n)

# function for counting the number of true and false positives
def count_tp_fp(selected_features, true_features):
	tp, fp = 0, 0
	for feature in selected_features:
		if feature in true_features:
			tp += 1
		else:
			fp += 1
	return tp, fp

#--------------------------------
# Run IPSS
#--------------------------------
ipss_output = ipss(X, y, selector='rf')

#--------------------------------
# Analyze results
#--------------------------------
runtime = ipss_output['runtime']
print(f'Runtime: {np.round(runtime,2)} seconds')
print(f'')

# select features based on target number of false positives
target_fp = 1
efp_scores = ipss_output['efp_scores']
selected_features = [idx for idx, efp_score in efp_scores.items() if efp_score <= target_fp]
tp, fp = count_tp_fp(selected_features, true_features)
print(f'-------- Target E(FP) = {target_fp} --------')
print(f'Selected features: {selected_features}')
print(f'Number of true positives: {tp}')
print(f'Number of false positives: {fp}')
print(f'')

# select features based on target FDR
target_fdr = 0.2
q_values = ipss_output['q_values']
selected_features = [idx for idx, q_value in q_values.items() if q_value <= target_fdr]
tp, fp = count_tp_fp(selected_features, true_features)
print(f'-------- Target FDR = {target_fdr} --------')
print(f'Selected features: {selected_features}')
print(f'Number of true positives: {tp}')
print(f'Number of false positives: {fp}')
print(f'')

#--------------------------------
# Plot stability paths
#--------------------------------
stability_paths = ipss_output['stability_paths']
n_alphas, p = stability_paths.shape

# blue paths for true features, gray for false features
color = ['dodgerblue' if i in true_features else 'gray' for i in range(p)]

for j in range(p):
	plt.plot(np.arange(n_alphas), stability_paths[:,j], color=color[j])
plt.tight_layout()
plt.show()



