import setuptools
from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="marquetry",
    version="0.0.1",
    license="MIT",
    install_requires=[
        "numpy",
        "pandas",
        "pillow",
    ],
    description="Simple Machine Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SHIMA",
    maintainer="SHIMA",
    author_email="shima@geeksheap.com",
    maintainer_email="shima@geeksheap.com",
    url="https://github.com/GeeksHeap/Marquetry",
    download_url="https://github.com/GeeksHeap/Marquetry",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    keywords="deeplearning ml neuralnetwork",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
