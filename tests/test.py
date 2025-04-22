# Basic functionality check 

from ipss import ipss
import numpy as np

def basic_test():
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

# main block to run the tests
if __name__ == '__main__':
	basic_test()
	print("All tests passed.")
