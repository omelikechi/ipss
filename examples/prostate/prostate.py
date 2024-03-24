# IPSS for prostate cancer
'''
Data freely available here: https://www.linkedomics.org/data_download/TCGA-PRAD/ 
'''

import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/omm793/iCloud/code/packages/ipss')
from ipss.ipss import ipss


#--------------------------------
# Load data
#--------------------------------
# select data type (reverse phase protein array (rppa) or microRNA (mirna))
data_type = 'mirna'
if data_type == 'rppa':
	data = np.load('prostate_rppa.npy', allow_pickle=True).item()
if data_type == 'mirna':
	data = np.load('prostate_mirna.npy', allow_pickle=True).item()

X, y, feature_names = data['X'], data['y'], data['feature_names']
n, p = X.shape
print(f'Data set contains {n} samples and {p} features')

#--------------------------------
# IPSS
#--------------------------------
result = ipss(X, y, EFP=1)

selected_features = result['selected_features']
print(f'\n-----Selected features-----')
for i in selected_features:
	print(f'Name (index): {feature_names[i]} ({i})')

stability_paths = result['stability_paths']
scores = result['scores']
threshold = result['threshold']

#--------------------------------
# Plot stability paths and scores
#--------------------------------
n_select = len(selected_features)
colormap = plt.cm.gist_rainbow
selected_colors = [colormap(i / n_select) for i in range(n_select)]

# define attributes for selected and non-selected features
selected_attr = {'alpha': 1, 'point_size': 100, 'line_width': 3}
non_selected_attr = {'alpha': 0.5, 'point_size': 50, 'line_width': 1.5}

# initialize lists for attributes
colors, alphas, point_sizes, line_widths = [], [], [], []

# assign attributes based on whether the feature is selected or not
j = 0
for i in range(p):
	if i in selected_features:
		color = selected_colors[j]
		j += 1
	else:
		color = 'gray'

	colors.append(color)
	alpha = selected_attr['alpha'] if i in selected_features else non_selected_attr['alpha']
	alphas.append(alpha)
	point_size = selected_attr['point_size'] if i in selected_features else non_selected_attr['point_size']
	point_sizes.append(point_size)
	line_width = selected_attr['line_width'] if i in selected_features else non_selected_attr['line_width']
	line_widths.append(line_width)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# plot stability paths
for i in range(p):
	ax[0].plot(stability_paths[:, i], color=colors[i], alpha=alphas[i], lw=line_widths[i],
			   label=feature_names[i] if i in selected_features else None)
ax[0].legend(loc='best')

# plot scores
noise = np.random.uniform(-0.5, 0.5, size=p)
ax[1].scatter(scores, noise, color=colors, alpha=alphas, s=point_sizes)
ax[1].axvline(threshold, linestyle='--', color='red', label=f'IPSS threshold')
ax[1].legend(loc='best')
ax[1].set_ylim(-1,1)

plt.tight_layout()
plt.show()







