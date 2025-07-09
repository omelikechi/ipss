from setuptools import setup, find_packages

setup(
	name='ipss',
	version='1.1.2',
	author='Omar Melikechi',
	author_email='omar.melikechi@gmail.com',
	packages=find_packages(),
	description='Python implementation of integrated path stability selection (IPSS)',
	long_description=open('README.md').read(),
	long_description_content_type='text/markdown',
	install_requires=[
		'joblib',
		'matplotlib',
		'numpy',
		'scikit-learn',
		'xgboost'
	],
	python_requires='>=3.6',
	include_package_data=True,
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
)
