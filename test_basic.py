import numpy as np
from ipss import ipss

# test 1: basic functionality check (continuous response)
def basic_test_continuous():
	# generate simple dataset
	X = np.random.normal(0, 1, size=(25, 50))
	y = np.sum(X[:,:5], axis=1) + np.random.normal(0, 1, size=25)

	# run ipss
	output = ipss(X,y)

	# check that output contains necessary keys
	assert 'efp_scores' in output, f"Missing 'efp_scores' in output"
	assert 'q_values' in output, f"Missing 'q_values' in output"
	assert 'runtime' in output, f"Missing 'runtime' in output"
	assert 'selected_features' in output, f"Missing 'selected_features' in output"
	assert 'stability_paths' in output, f"Missing 'stability_paths' in output"

# test 2: basic functionality check (binary response)
def basic_test_binary():
	# generate simple dataset
	X = np.random.normal(0, 1, size=(25, 50))
	signal = np.sum(X[:,:5], axis=1) + np.random.normal(0, 1, size=25)
	y = (signal > np.median(signal)).astype(int)

	# run ipss
	output = ipss(X,y)

	# check that output contains necessary keys
	assert 'efp_scores' in output, f"Missing 'efp_scores' in output"
	assert 'q_values' in output, f"Missing 'q_values' in output"
	assert 'runtime' in output, f"Missing 'runtime' in output"
	assert 'selected_features' in output, f"Missing 'selected_features' in output"
	assert 'stability_paths' in output, f"Missing 'stability_paths' in output"

# main block to run the tests
if __name__ == '__main__':
	basic_test_continuous()
	basic_test_binary()
	print("All tests passed.")
