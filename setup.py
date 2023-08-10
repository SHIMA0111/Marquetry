from setuptools import setup

setup(
    name="matrixml",
    version="0.0.1",
    license="MIT",
    install_requires=[
        "numpy",
        "pandas",
        "pillow"
    ],
    description="Simple Machine Learning Framework",
    author="SHIMA",
    author_email="shima@geeksheap.com",
    url="",
    packages=["matrixml"],
    python_requires=">=3.6"
)