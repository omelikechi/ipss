# IPSS for colon cancer
'''
Data freely available here: http://genomics-pubs.princeton.edu/oncology/affydata/index.html
'''

import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append('/Users/omm793/iCloud/code/packages/ipss')
from ipss.ipss import ipss


#--------------------------------
# Load data
#--------------------------------
data = np.load('colon_data.npy', allow_pickle=True).item()
X, y, feature_names = data['X'], data['y'], data['feature_names']
n, p = X.shape
print(f'Data set contains {n} samples and {p} features')

# order by cancerous first, then normal
sorted_indices = np.argsort(y)
y = y[sorted_indices] 
X = X[sorted_indices] 

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

#--------------------------------
# Plot heatmap
#--------------------------------
names = [feature_names[i] for i in selected_features]

X = X[:,selected_features]

n_normal = np.sum(y)
n_cancerous = n - n_normal
print(n_normal)
print(n_cancerous)

avg_expression_cancerous = []
for i in range(len(selected_features)):
    avg_expression_cancerous.append(np.mean(X[:n_cancerous,i]))

sorted_indices = np.argsort(avg_expression_cancerous)
X = X[:,sorted_indices]
names = [names[i] for i in sorted_indices]

# plot the heatmap
plt.figure(figsize=(8,6))
sns.heatmap(X, cmap='binary')

# draw vertical lines to separate the regions
plt.axhline(y=n_cancerous, color='red', linestyle='--', linewidth=3)

# annotate the regions with labels
plt.text(-1/2, n_cancerous / 2, 'Cancerous', ha='center', va='center', rotation='vertical', fontsize=16)
plt.text(-1/2, n_cancerous + n_normal / 2, 'Normal', ha='center', va='center', rotation='vertical', fontsize=16)

# set tick rotation for x-axis
plt.xticks(np.arange(len(names)), names, rotation=45, ha='left')
for tick, color in zip(plt.gca().get_xticklabels(), colors):
    tick.set_color(color)
    tick.set_weight('bold')
plt.yticks([])

plt.title('Expression levels of selected genes', fontsize=16)
plt.tight_layout()
plt.show()







