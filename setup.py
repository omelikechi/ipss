from setuptools import setup, find_packages

setup(
    name='ipss',
    version='0.4.1',
    author='Omar Melikechi',
    author_email='omar.melikechi@gmail.com',
    packages=find_packages(),
    description='Python implementation of Integrated Path Stability Selection (IPSS)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'joblib',
        'seaborn'
    ],
    python_requires='>=3.6',
    include_package_data=True,
    classifiers=[
        # Classifiers help users find your project
        # Full list: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
