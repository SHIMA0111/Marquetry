import setuptools
from setuptools import setup

long_description = """# Marquetry
Marquetry means **Yosegi-zaiku**, a traditional Japanese woodworking technique, in Japan.  
It is a beautiful culture craft originated in Japan, which is a box or ornament or so by small wooden pieces.  
The design is UNIQUE, it depends on the arrangement of the wood tips.  
I believe Deep Learning is similar with the concept.  
Deep Learning models are constructed through the combination of the layers or functions. 
Just as a slight variation in arrangement can result in a completely distinct model.  
I want you can enjoy the deep/machine learning journey like 
you craft a **Marquetry** from combination of various wood tips. 

## About this Framework
You can use this framework for help your learning **Machine/Deep Learning**.  
This framework is written only **Python**, so you can understand the implementation easily if you are python engineer.  
For simplify the construct, there are un-efficiency implementation.  
I develop this framework to enjoy learning the construction of the machine/machine learning not **Practical Usage**.  
I hope to enjoy your journey!"""

setup(
    name="marquetry",
    version="0.0.2",
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
