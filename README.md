## Simple Machine Learning Framework
You can use this framework for your learning **Machine Learning**/**Deep Learning**.  
This framework is written only **Python**, so you can understand the implementation easily if you are python engineer.

### Directory
```
├── README.md
├── gradtracer
│   ├── __init__.py
│   ├── core.py ... Core components
│   ├── datasets.py ... Dataset like "MNIST"/"Titanic"
│   ├── functions.py ... Functions for layer/model construction
│   ├── layers.py ... Layers conponents
│   ├── models.py ... Example models
│   ├── optimizers.py ... Model optimizer
│   └── utils.py ... other utils
├── setup.py
└── tests

```

### Dependencies
You need to fill the below version requirement and import external libraries. 

 - [Python 3 | 3.6 or later](https://docs.python.org/3/)
 - [NumPy](https://numpy.org/)
 - [Pandas](https://pandas.pydata.org/)

for display the calculation graph
 - [Pillow](https://pillow.readthedocs.io/en/stable/)

for test script
 - [PyTorch](https://pytorch.org/)

### Reference Source
This framework refer to [dezero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).  
Therefore, there are much similar architecture between **dezero** and this like the algorithm of the autograd and so.
If you want to know about this framework deeply, I suggest to visit the **dezero** repository.
