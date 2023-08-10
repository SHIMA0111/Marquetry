from setuptools import setup

setup(
    name="gradtracer",
    version="0.0.1",
    license="MIT",
    install_requires=[
        "numpy",
        "pandas",
        "pillow",
        "torch"
    ],
    description="Simple Machine Learning Framework",
    author="SHIMA",
    author_email="shima@geeksheap.com",
    url="",
    packages=["gradtracer"],
    python_requires=">=3.8"
)