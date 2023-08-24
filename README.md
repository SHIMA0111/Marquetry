# Marquetry
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
I hope to enjoy your journey!  

### Directory
```
├── README.md
├── marquetry
│   ├── __init__.py
│   ├── pre_implementation ... preview implement components
│   │   └── svm.py
│   ├── conventional_ml ... conventional machine learning components
│   │   └── tree.py ... DecisionTree and RandomForest
│   │
│   ├── core.py ... Core components of the marquetry
│   ├── datasets.py ... Dataset like "MNIST"/"Titanic" and so
│   ├── dataloaders.py ... Dataloader components
│   ├── functions.py ... Functions for layer/model construction
│   ├── cuda_backend.py ... using CuPy component and transform NumPy and CuPy component
│   ├── layers.py ... Layers conponents
│   ├── models.py ... Example models
│   ├── optimizers.py ... Model optimizer
│   ├── preprocess.py ... Preprocess Script
│   ├── transformers.py ... Data transformers
│   └── utils.py ... other utils
├── setup.py
└── tests

```

### Dependencies
You need to fill the below version requirement and import external libraries. 

 - [Python 3 | 3.8 or later](https://docs.python.org/3/)
 - [NumPy](https://numpy.org/)
 - [Pandas](https://pandas.pydata.org/)

Optional
 - [CyPy](https://cupy.dev/)

for display the calculation graph
 - [Pillow](https://pillow.readthedocs.io/en/stable/)

for test script
 - [PyTorch](https://pytorch.org/)


### License
This project is licensed under the [MIT License](LICENSE.md).


### Reference Source
This framework started to be developed based on [dezero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).  
Originally, the dezero was developed based on **[Chainer](https://tutorials.chainer.org/ja/)**
(and **[PyTorch](https://pytorch.org/)**) architecture.   
Therefore, there are much similar architecture between **dezero**(**Chainer**) and 
this like the algorithm of the auto-gradient and so.  
If you want to know about this framework deeply, I suggest to visit the **dezero** and **Chainer** repository. 
(PyTorch is more complex but beautiful.)

And, I respect the **dezero** author and his books are very curiously and easy to understandable.  
If you want to start journey for deep learning world, I suggest to read [his books](https://www.oreilly.co.jp/books/9784873117584/).  
