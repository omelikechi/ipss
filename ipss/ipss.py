# Integrate path stability selection (IPSS)

import warnings

from joblib import Parallel, delayed
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import lars_path, Lasso, lasso_path, LogisticRegression
from sklearn.preprocessing import StandardScaler


#--------------------------------
# IPSS regression
#--------------------------------
'''
Inputs:
	X: n-by-p data matrix (n = number of samples, p = number of features)
	y: n-by-1 response vector (binary or continuous)
	EFP: Target value for expected number of false positives
	cutoff: Positive scalar C that, together with EFP, determines IPSS threshold
	B: Number of subsampling steps
	n_alphas: Number of values in grid of regularization parameters
	q_max: Maximum number of features selected
	Z_sparse: n_alphas-by-B-by-p tensor of subsamples, Z, is output as sparse if 'True'
	lars: Uses least angle regression (LARS) for linear regression if 'True' or lasso if 'False'
	selection_function: Function to apply to the estimated selection probabilities. If equal to
		an integer, m, then function is h_m(x) = (2x - 1)**m if x >= 0.5 and 0 if x < 0.5
	with_stability: Uses stability measure if 'True'
	delta: Scalar value that determines scaling of regularization interval. delta = 1 corresponds
		to log scale, delta = 0 corresponds to linear scale
'''
def ipss(X, y,
	EFP=1,
	cutoff=0.05, 
	B=50, 
	n_alphas=25,
	q_max=None, 
	Z_sparse=False, 
	lars=False,
	selection_function=None,
	with_stability=False,
	delta=1,
	standardize_X=True,
	center_y=True
	):

	if len(y.shape) != 1:
		if y.shape[1] == 1:
			y = y.ravel()
		else:
			raise ValueError("Error: Response y must be a numpy array with shape (n,) or (n,1)")

	if standardize_X:
		X = StandardScaler().fit_transform(X)

	n, p = X.shape
	n_split = int(n/2)

	# check if response is binary
	binary_response = (len(np.unique(y)) == 2)

	# maximum number of features
	if q_max is None:
		q_max = p / 2

	# compute alphas
	alphas = compute_alphas(X, y, n_alphas, q_max, binary_response)

	# linear regression
	if not binary_response:

		if center_y:
			y -= np.mean(y)

		if lars:
			def process_b(b):
				indices = np.arange(n)
				np.random.shuffle(indices)

				z = np.empty((n_alphas, 2, p))

				for half in range(2):
					idx = indices[:n_split] if half == 0 else indices[n_split:]
					X_half, y_half = X[idx,:], y[idx]

					with warnings.catch_warnings():
						warnings.simplefilter('ignore')
						lars_alphas, _, coefs = lars_path(X_half, y_half, method='lasso')

					for i, alpha in enumerate(alphas):
						idx_alpha = np.abs(lars_alphas - alpha).argmin()
						coef = coefs[:, idx_alpha]

						z[i, half, :] = (coef != 0).astype(int)

				return z

		else:
			def process_b(b):
				indices = np.arange(n)
				np.random.shuffle(indices)

				z = np.empty((n_alphas, 2, p))

				for half in range(2):
					idx = indices[:n_split] if half == 0 else indices[n_split:]
					X_half, y_half = X[idx,:], y[idx]

					with warnings.catch_warnings():
						warnings.simplefilter('ignore')
						_, coef, _ = lasso_path(X_half, y_half, alphas=alphas)
					z[:, half, :] = (coef.T != 0).astype(int)

				return z

	# logistic regression
	else:
		def process_b(b):
			indices = np.arange(n)
			np.random.shuffle(indices)

			z = np.empty((n_alphas, 2, p))

			model = LogisticRegression(penalty='l1', solver='saga', tol=1e-3, warm_start=True, class_weight='balanced')
			# model = LogisticRegression(penalty='l1', max_iter=int(1e6), solver='liblinear', class_weight='balanced')

			for half in range(2):
				idx = indices[:n_split] if half == 0 else indices[n_split:]
				X_half, y_half = X[idx,:], y[idx]

				for i, alpha in enumerate(alphas):

					model.set_params(C=1/alpha)
					with warnings.catch_warnings(record=True) as w:
						warnings.simplefilter('ignore')
						fit = model.fit(X_half, y_half.ravel())
					z[i, half, :] = (fit.coef_ != 0).astype(int)

			return z

	# parallelize subsampling across multiple CPU cores
	results = np.array(Parallel(n_jobs=-1)(delayed(process_b)(b) for b in range(B)))

	# aggregate results
	Z = np.zeros((n_alphas,2*B,p))
	for b in range(B):
		Z[:, 2*b:2*(b + 1), :] = results[b,:,:,:]

	# stop at max features, q_max
	stop_index = n_alphas
	average_select = np.empty(n_alphas)
	for i in range(n_alphas):
		z = Z[i,:,:]
		average_select[i] = np.mean(np.sum(z,axis=1))
		if average_select[i] > q_max:
			stop_index = i
			break

	Z = Z[:stop_index,:,:]
	alphas = alphas[:stop_index]
	average_select = average_select[:stop_index]

	if Z_sparse:
		Z_sparse = np.empty((stop_index,), dtype=object)
		for i in range(stop_index):
			Z_sparse[i] = csr_matrix(Z[i,:,:])
		Z = Z_sparse

	# ipss
	if selection_function is None:
		if binary_response:
			selection_function = 2
		else:
			selection_function = 3

	stability_paths, scores, integral, alphas, stop_index = ipss_results(Z, alphas, average_select, selection_function, with_stability, delta, cutoff)

	threshold = integral / EFP
	selected_features = np.where(scores >= threshold)[0]


	return {'alphas':alphas, 'average_select':average_select, 'scores':scores, 'selected_features':selected_features, 
		'stability_paths':stability_paths, 'stop_index':stop_index, 'threshold':threshold}


#--------------------------------
# IPSS scores
#--------------------------------
def ipss_results(Z, alphas, average_select, selection_function, with_stability, delta, cutoff):

	n_alphas = Z.shape[0]
	B, p = Z[0].shape
	B /= 2

	# function
	if isinstance(selection_function, (int, float)):
		m = selection_function
		def selection_function(x):
			return 0 if x <= 0.5 else (2*x - 1)**m
	else:
		m = 'user_defined'

	# stability paths
	stability_paths = np.empty((n_alphas,p))
	for i in range(n_alphas):
		stability_paths[i] = Z[i].mean(axis=0)

	# stability measure
	if with_stability:
		stability_values = np.array([stability(Z[i]) for i in range(n_alphas)])
		normalizer, _ = integrate(stability_values, alphas, delta)
		stability_values /= normalizer
	else:
		stability_values = np.ones(n_alphas)

	# evaluate ipss bounds for specific functions
	if m == 1:
		integral, stop_index = integrate(stability_values * average_select**2 / p, alphas, delta, cutoff=cutoff)
	elif m == 2:
		term1 = average_select**2 / (p * B)
		term2 = (B-1) * average_select**4 / (B * p**3)
		integral, stop_index  = integrate(stability_values * (term1 + term2), alphas, delta, cutoff=cutoff)
	elif m == 3:
		term1 = average_select**2 / (p * B**2)
		term2 = (3 * (B-1) * average_select**4) / (p**3 * B**2)
		term3 = ((B-1) * (B-2) * average_select**6) / (p**5 * B**2)
		integral, stop_index = integrate(stability_values * (term1 + term2 + term3), alphas, delta, cutoff=cutoff)
	else:
		integral = cutoff
		stop_index = len(alphas)

	# compute ipss scores
	alphas_stop = alphas[:stop_index]
	scores = np.zeros(p)
	for i in range(p):
		values = np.empty(stop_index)
		for j in range(stop_index):
			values[j] = stability_values[j] * selection_function(stability_paths[j,i])
		scores[i], _ = integrate(values, alphas_stop, delta)


	return stability_paths, scores, integral, alphas, stop_index


#--------------------------------
# Helpers
#--------------------------------
def compute_alphas(X, y, n_alphas, q_max, binary_response=False):
	n, p = X.shape

	if binary_response:
		y_mean = np.mean(y)
		scaled_residuals = y - y_mean * (1 - y_mean)
		alpha_max = 5 / np.max(np.abs(np.dot(X.T, scaled_residuals) / n))
		model = LogisticRegression(penalty='l1', solver='saga', tol=1e-3, warm_start=True, class_weight='balanced')
		# model = LogisticRegression(penalty='l1', max_iter=int(1e6), solver='liblinear', class_weight='balanced')
	else:
		alpha_max = 2 * np.max(np.abs(np.dot(X.T,y))) / n
		model = Lasso(warm_start=True)

	alpha_min = alpha_max * 1e-10
	test_alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), 100)

	for i, alpha in enumerate(test_alphas):
		if binary_response:
			model.set_params(C=1/alpha)
		else:
			model.set_params(alpha=alpha)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			model.fit(X,y)
		num_selected = np.sum(model.coef_ != 0)
		if num_selected >= q_max:
			alpha_min = alpha
			break

	alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)

	return alphas


def integrate(values, alphas, delta=1, cutoff=None):

	n_alphas = len(alphas)
	a = min(alphas)
	b = max(alphas)

	if delta == 1:
		normalization = (1 - (a/b)**(1/n_alphas)) / np.log(b/a)
	else:
		normalization = (1 - delta) * (1 - (a/b)**(1/n_alphas)) / (b**(1-delta) - a**(1-delta))

	output = 0
	stop_index = n_alphas
	before = stop_index

	if cutoff is None:
		for i in range(1,n_alphas):
			weight = 1 if delta == 1 else alphas[i]**(1-delta)
			output += normalization * weight * values[i-1]

	else:
		for i in range(1,n_alphas):
			weight = 1 if delta == 1 else alphas[i]**(1-delta)
			updated_output = output + normalization * weight * values[i-1]
			if updated_output > cutoff:
				stop_index = i
				break
			else:
				output = updated_output

	return output, stop_index


def stability(z):
	B, d = np.shape(z)
	prob = np.mean(z,axis=0)
	prob = np.squeeze(np.asarray(prob))
	k_hat = np.mean(prob)
	numerator = np.mean(prob * (1 - prob))
	denominator = k_hat * (1 - k_hat)
	if denominator > 1e-8:
		frac = numerator/denominator
	else:
		frac = 1

	return 1 - frac
















