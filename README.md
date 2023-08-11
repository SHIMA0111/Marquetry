# Marquetry
Marquetry means **Yosegi-zaiku** in Japan.  
It is a beautiful culture craft originated in Japan, which is a box or ornament or so by small wood tips.  
The design is UNIQUE because the signature depends on the combination of the wood tips. 
I believe Deep Learning is similar with the concept. 
Deep Learning models are created by the combination of the layers or functions. 
Even if a model constructs the same components as another but the combination(order) is different, 
these are different completely.
I want you can enjoy the deep/machine learning journey like 
you craft a **Marquetry** from combination of various wood tips. 


## Simple Machine Learning Framework
You can use this framework for your learning **Machine Learning**/**Deep Learning**.  
This framework is written only **Python**, so you can understand the implementation easily if you are python engineer.  
For simplify the construct, there are un-efficiency implementation.  
I develop this framework to enjoy learning the construction of the machine/machine learning not **Practical Use**.  
I hope to enjoy your journey!  

### Directory
```
├── README.md
├── marquetry
│   ├── __init__.py
│   ├── core.py ... Core components of the marquetry
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
This framework started to be developed based on [dezero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).  
Therefore, there are much similar architecture between **dezero** and this like the algorithm of the autograd and so.  
If you want to know about this framework deeply, I suggest to visit the **dezero** repository.

I respect the author because his books are very curiously and easy to understandable.  
If you want to start journey for deep learning world, I suggest to read [his books](https://www.oreilly.co.jp/books/9784873117584/).  
