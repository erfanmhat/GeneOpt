from setuptools import setup, find_packages

with open("README.md", "r") as file:
    description = file.read()

setup(
    name="GeneOpt",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "catboost>=1.0.0",
        "scikit-learn>=0.24.0"
    ],
    long_description=description,
    long_description_content_type="text/markdown",
)
