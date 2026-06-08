from setuptools import setup, find_packages

setup(
    name='vibro-ml',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pyyaml',
    ],
)
