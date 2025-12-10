from setuptools import setup, find_packages

setup(
    name="dis_spatial",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "scipy",
        "scikit-learn",
        "networkx",
        "numpy",
        "matplotlib",
        "joblib",
    ],
    python_requires=">=3.6",
    description="decentralized spatial statistics package for low-rank spatial models",
)
