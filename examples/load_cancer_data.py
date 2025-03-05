# Functions for loading LinkedOmics data

import os

import numpy as np
import pandas as pd

def load_data(cancer_type, feature_types, response=None, remove_nan=True, correlation_threshold=0.999, see_names=False, verbose=False):

	base_path = os.path.join(os.path.dirname(__file__), cancer_type + '_data')
	
	file_paths = {
		'clinical': os.path.join(base_path, 'clinical.txt'),
		'methylation': os.path.join(base_path, 'methylation.txt'),
		'mirna': os.path.join(base_path, 'mirna.txt'),
		'mutation': os.path.join(base_path, 'mutation.txt'),
		'rnaseq': os.path.join(base_path, 'rnaseq.txt'),
		'rppa': os.path.join(base_path, 'rppa.txt')
	}

	# initialize shared labels
	shared_labels = None

	# response
	if response is not None:
		response_type, response_name = response 
		response_df = load_dataframe(file_paths[response_type])
		Y_df = response_df[response_name].apply(pd.to_numeric, errors='coerce')
		shared_labels = set(Y_df.index)
	else:
		response_type = None

	# features
	feature_dfs = []
	feature_names = []
	for category in feature_types:
		df = load_dataframe(file_paths[category])
		if category == response_type:
			# remove the response variable from the feature dataframe
			df = df.drop(columns=[response_name], errors='ignore')
		if shared_labels is not None:
			shared_labels = shared_labels.intersection(df.index)
		else:
			shared_labels = set(df.index)
		feature_dfs.append(df)
		feature_names.extend(df.columns.tolist())
		if category == 'mirna':
			feature_names = [mirna.replace('hsa-mir-', 'miR-') for mirna in feature_names]
			feature_names = [mirna.replace('hsa-let-', 'let-') for mirna in feature_names]


	# ensure there are shared labels
	if not shared_labels:
		raise ValueError("No shared labels found across the feature datasets.")

	# only keep shared labels
	shared_labels = sorted(shared_labels)
	feature_dfs = [df.loc[shared_labels] for df in feature_dfs]

	if verbose:
		print(f'Number of rows with shared labels: {len(shared_labels)}')

	if see_names:
		print(feature_names)
		return

	# concatenate features into X
	X = pd.concat(feature_dfs, axis=1).to_numpy()
	# print(X)
	X = X.astype(float)

	# remove columns with NaN values
	if remove_nan:
		cols_with_nan = np.isnan(X).any(axis=0)
		X = X[:, ~cols_with_nan]
		feature_names = [name for (name, keep) in zip(feature_names, ~cols_with_nan) if keep]

	# remove highly correlated columns (correlation > correlation_threshold)
	if X.shape[1] > 1:  # Check if there is more than one column
		correlation_matrix = np.corrcoef(X, rowvar=False)

		# identify pairs of columns with correlation > threshold
		correlated_columns = set()
		for i in range(len(correlation_matrix)):
			for j in range(i+1, len(correlation_matrix)):
				if abs(correlation_matrix[i, j]) > correlation_threshold:
					correlated_columns.add(j)

		# keep uncorrelated columns
		uncorrelated_columns = [i for i in range(X.shape[1]) if i not in correlated_columns]
		X = X[:, uncorrelated_columns]
		feature_names = [feature_names[i] for i in uncorrelated_columns]

		if verbose:
			print(f'Removed {len(correlated_columns)} columns due to high correlation.')

	if response is not None:
		Y = Y_df.loc[shared_labels].to_numpy()
		rows_with_nan = np.isnan(Y)
		X = X[~rows_with_nan,:]
		Y = Y[~rows_with_nan]
		return {'X':X, 'Y':Y, 'feature_names':feature_names, 'response':response}
	else:
		return {'X':X, 'Y':None, 'feature_names':feature_names, 'response':response}

# load dataframes
def load_dataframe(file_path):
	df = pd.read_csv(file_path, sep='\t').T
	df.columns = df.iloc[0]
	df = df.drop(df.index[0])
	df.index.name = 'attrib_name'
	return df



