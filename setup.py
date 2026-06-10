import setuptools
from setuptools import setup

import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="marquetry",
    version="0.3.0",
    license="MIT",
    install_requires=[
        # Floors are the first NumPy-2-compatible line of each dependency;
        # the framework itself assumes NumPy 2 (NEP 50) promotion semantics.
        "numpy>=2.0.0",
        "pandas>=2.2.0",
        "Pillow>=10.4.0",
        "scipy>=1.13.0"
    ],
    extras_require={
        "onnx": ["onnx>=1.16.0"],
    },
    description="Pure Python Deep/Machine Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SHIMA0111",
    maintainer="SHIMA0111",
    author_email="shima@little-tabby.com",
    maintainer_email="engineer@little-tabby.com",
    url="https://github.com/SHIMA0111/Marquetry",
    download_url="https://github.com/SHIMA0111/Marquetry",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    keywords="deeplearning ml neuralnetwork",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
