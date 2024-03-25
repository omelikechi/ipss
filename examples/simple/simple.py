# A simple example

import matplotlib.pyplot as plt
import numpy as np

from ipss import ipss


#--------------------------------
# Simulate data
#--------------------------------
# data parameters
n = 100  # number of samples
p = 200  # number of features
snr = 2  # signal-to-noise ratio

# generate truth
n_signals = 10  # number of "true" features
indices = np.random.choice(np.arange(p), size=n_signals, replace=False)  # randomly select true indices
beta = np.zeros(p)
beta[indices] = np.random.normal(0,1,size=n_signals)
truth = (beta != 0).astype(int)

# generate data
X = np.random.normal(0, 1, size=(n,p))
signal = X @ beta
sigma = np.sqrt(np.var(signal) / snr)  # noise
y = signal + np.random.normal(0, sigma, size=n)

#--------------------------------
# IPSS
#--------------------------------
result = ipss(X, y, EFP=1)

stability_paths = result['stability_paths']
scores = result['scores']
threshold = result['threshold']

#--------------------------------
# Plot results
#--------------------------------
# specify plot details
color = ['dodgerblue' if value == 1 else 'gray' for value in truth]
alpha = [1 if value == 1 else 0.5 for value in truth]
point_size = [100 if value == 1 else 75 for value in truth]

fig, ax = plt.subplots(1, 2, figsize=(14,5))

# plot stability paths
for j in range(p):
	ax[0].plot(stability_paths[:,j], color=color[j], alpha=alpha[j])
ax[0].set_title(f'Stability paths', fontsize=16)

# plot scores
noise = np.random.uniform(-.5, .5, size=p)
ax[1].scatter(scores, noise, color=color, alpha=alpha, s=point_size)
ax[1].axvline(threshold, linestyle='--', color='red', label=f'IPSS threshold')
ax[1].legend(loc='best', fontsize=14)
ax[1].set_ylim(-1,1)
ax[1].set_title(f'IPSS scores', fontsize=16)

plt.tight_layout()
plt.show()










