from setuptools import setup, find_packages

setup(
    name="gist",
    version="0.1.0",
    description="GIST: Greedy Independent Set Thresholding for Max-Min Diversification",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.9",
)
