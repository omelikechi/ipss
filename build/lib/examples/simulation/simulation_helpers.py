# Helper functions for the simulation experiments

import numpy as np
from sklearn.preprocessing import StandardScaler

# function for generating features
def generate_features(n, p, feature_type='standard_normal', rho=None, standardize=True):
	if feature_type == 'standard_normal':
		X = np.random.normal(0, 1, size=(n,p))
	elif feature_type == 'toeplitz':
		if rho is None or not (0 < rho < 1):
			raise ValueError("A rho between 0 and 1 must be specified for the Toeplitz feature type.")
		Sigma = np.empty((p,p))
		for i in range(p):
			for j in range(p):
				Sigma[i,j] = rho**np.abs(i-j)
		X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
	elif feature_type == 'ovarian_rnaseq':
		X_full = np.load('ovarian_rnaseq.npy')
		n_full, p_full = X_full.shape
		samples = np.random.choice(n_full, size=n, replace=False)
		features = np.random.choice(p_full, size=p, replace=False)
		X = X_full[:,features]
		X = X[samples,:]
	else:
		raise ValueError(f"Unknown feature_type: {feature_type}")

	if standardize:
		X = StandardScaler().fit_transform(X)
	return X

# function for generating response
def generate_response(X, n_true, snr, nonlinear=False, response_type='continuous'):

	# dimensions
	n, p = X.shape

	# randomly select true features
	true_features = np.random.choice(p, n_true, replace=False)

	# signal
	if nonlinear:

		function_list = [
			lambda x, alpha, beta: (1 + np.tanh((alpha/2) * (x - beta))) / 2,
			lambda x, alpha, beta: np.exp(-alpha * x**2),
		]
		alpha_range = [1, 3]
		beta_range = [-1, 1]
		n_func = len(function_list)

		# generate partitions
		n_partitions = np.random.choice(np.arange(int(n_true / 2), n_true + 1))
		array = np.arange(n_true)
		np.random.shuffle(array)
		split_points = np.sort(np.random.choice(range(1, n_true), n_partitions - 1, replace=False))
		partitions = np.split(array, split_points)
		partitions = [part.tolist() for part in partitions]

		# apply functions to partitioned features
		signal = np.zeros(n)
		sums = np.zeros((n, n_partitions))
		for i, partition in enumerate(partitions):
			for j in partition:
				sums[:,i] += X[:,true_features[j]]
			sums[:,i] -= np.mean(sums[:,i])
			sums[:,i] /= np.std(sums[:,i])
			sums[:,i] *= np.random.choice([-1,1])
			f = function_list[np.random.choice(n_func)]
			alpha = np.random.uniform(alpha_range[0], alpha_range[1])
			beta = np.random.uniform(beta_range[0], beta_range[1])
			signal += np.random.choice([-1,1]) * f(sums[:,i], alpha, beta)
	else:
		beta = np.zeros(p)
		beta[true_features] = np.random.normal(0, 1, size=n_true)
		signal = X @ beta


	# add noise to signal to get response
	if response_type == 'continuous':
		sigma = np.sqrt(np.var(signal) / snr)
		y = signal + np.random.normal(0, sigma, size=n)
		y -= np.mean(y)
		y /= np.std(y)
	else:
		signal -= np.mean(signal)
		prob = 1 / (1 + np.exp(-snr * signal))
		y = np.random.binomial(1, prob, size=n)

	return y, true_features








