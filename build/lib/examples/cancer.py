# Analyze human cancer data from The Cancer Genome Atlas via LinkedOmics
# source: https://www.linkedomics.org/login.php

import matplotlib.pyplot as plt

from ipss import ipss
from load_cancer_data import load_data

#--------------------------------
# Available data options
#--------------------------------
"""
Available options for the different data types are:
cancer type: 'ovarian' or 'prostate'
feature type: 'clinical', 'mirna', and/or 'rppa'
response type: tuple of strings (feature_type, feature_name), where
	- feature_type is any one of the feature types above
	- feature_name is any feature name in the specified feature type, e.g., 'Tumor_purity' in 'clinical'

Original data freely available here:
	- ovarian cancer: https://www.linkedomics.org/data_download/TCGA-OV/
	- prostate cancer: https://www.linkedomics.org/data_download/TCGA-PRAD/
"""
# uncomment both lines below to print all feature names for a given feature type (e.g., to see response variable options)
# cancer_type, feature_types = 'ovarian', ['clinical']
# data = load_data(cancer_type, feature_types, see_names=True)

#--------------------------------
# Load data
#--------------------------------
cancer_type = 'ovarian'
feature_types = ['mirna']
response = ('clinical', 'Tumor_purity')

data = load_data(cancer_type, feature_types, response=response)
X, y, feature_names = data['X'], data['Y'], data['feature_names']

#--------------------------------
# Run IPSS
#--------------------------------
ipss_output = ipss(X, y)

#--------------------------------
# Analyze results
#--------------------------------
plot_q_values = True
plot_efp_scores = True

# plot q-values for all features with q-value below a certain threshold
if plot_q_values:
	q_value_threshold = 0.5
	q_values = ipss_output['q_values']
	filtered_q_values = {idx: q_value for idx, q_value in q_values.items() if q_value <= q_value_threshold}
	sorted_q_values = dict(sorted(filtered_q_values.items(), key=lambda x: x[1]))

	plt.figure(figsize=(10, 6))
	plt.bar(range(len(sorted_q_values)), sorted_q_values.values(), color='dodgerblue')
	plt.xticks(range(len(sorted_q_values)), [feature_names[idx] for idx in sorted_q_values], rotation=45)
	plt.ylabel('$q$-value', fontsize=18)
	plt.tight_layout()
	plt.show()

# plot efp scores for all features with efp scores below a certain threshold
if plot_efp_scores:
	efp_score_threshold = 5
	efp_scores = ipss_output['efp_scores']
	filtered_efp_scores = {idx: efp_score for idx, efp_score in efp_scores.items() if efp_score <= efp_score_threshold}
	sorted_efp_scores = dict(sorted(filtered_efp_scores.items(), key=lambda x: x[1]))

	plt.figure(figsize=(10, 6))
	plt.bar(range(len(sorted_efp_scores)), sorted_efp_scores.values(), color='dodgerblue')
	plt.xticks(range(len(sorted_efp_scores)), [feature_names[idx] for idx in sorted_efp_scores], rotation=45)
	plt.ylabel('efp score', fontsize=18)
	plt.tight_layout()
	plt.show()











