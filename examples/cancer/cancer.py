# Analyze human cancer data from The Cancer Genome Atlas via LinkedOmics
# source: https://www.linkedomics.org/login.php

import matplotlib.pyplot as plt

from ipss import ipss
# from load_cancer_data import load_data
from examples.cancer.load_cancer_data import load_data

#--------------------------------
# Available data options
#--------------------------------
"""
Available options for the different data types are:
cancer type: 'ovarian' or 'prostate'
feature type: 'clinical', 'methylation', 'mirna', 'rnaseq', and/or 'rppa'
response type: None if no response, else tuple of strings (feature_type, feature_name), where
	- feature_type is any one of the feature types above
	- feature_name is any feature name in the specified feature type, e.g., 'Tumor_purity' in 'clinical'

For further details about the specific datasets:
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
feature_types = ['rnaseq']
response = ('clinical', 'status')

data = load_data(cancer_type, feature_types, response=response)
X, y, feature_names = data['X'], data['Y'], data['feature_names']

#--------------------------------
# Run IPSS
#--------------------------------
ipss_output = ipss(X, y, selector='gb')

#--------------------------------
# Analyze results
#--------------------------------
plot_q_values = True
plot_efp_scores = True

# plot q-values for all features with q-values below a certain threshold
if plot_q_values:
	q_value_threshold = 0.5
	q_values = ipss_output['q_values']
	q_values = [(index, q_value) for (index, q_value) in q_values if q_value <= q_value_threshold]

	plt.figure(figsize=(10, 6))
	for i, (feature_index, q_value) in enumerate(q_values):
		plt.bar(i, q_value, color='dodgerblue')
	plt.xticks(range(len(q_values)), [feature_names[feature_index] for feature_index, _ in q_values], rotation=45)
	plt.ylabel('$q$-value', fontsize=18)
	plt.tight_layout()
	plt.show()

# plot efp scores for all features with efp scores below a certain threshold
if plot_efp_scores:
	efp_score_threshold = 5
	efp_scores = ipss_output['efp_scores']
	efp_scores = [(index, q_value) for (index, q_value) in efp_scores if q_value <= efp_score_threshold]

	plt.figure(figsize=(10, 6))
	for i, (feature_index, efp_score) in enumerate(efp_scores):
		plt.bar(i, efp_score, color='dodgerblue')
	plt.xticks(range(len(efp_scores)), [feature_names[feature_index] for feature_index, _ in efp_scores], rotation=45)
	plt.ylabel('efp score', fontsize=18)
	plt.tight_layout()
	plt.show()











