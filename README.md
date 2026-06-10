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
For simplify the construct, there are un-efficiency implementations.
I develop this framework to enjoy learning the construction of the machine/deep learning not **Practical Usage**.
I hope to enjoy your journey!

## Features
 - **Define-by-run automatic differentiation** — the computation graph is recorded
   while your Python code runs, so any control flow just works
 - **Neural network building blocks** — layers (Linear, Convolution2D, Deconvolution2D,
   BatchNormalization, BatchRenormalization, LayerNormalization, RNN/LSTM/BiLSTM/GRU,
   Embedding, ...),
   70+ functions (activations, losses, evaluation metrics, ...), and ready-made models
   (MLP, CNN, Sequential)
 - **Optimizers** — SGD, MomentumSGD, Nesterov, AdaGrad, AdaDelta, RMSProp,
   Adam, AdamW, AdaMax, Nadam, and Lion
 - **Data utilities** — built-in datasets (FashionMNIST, Titanic, synthetic toy sets),
   DataLoader / SeqDataLoader, transformers, and pandas-based tabular preprocessing
   (encoding, imputation, normalization)
 - **Classic machine learning** — decision trees, random forest, and SVM,
   implemented from scratch with the same `fit` / `predict` interface
 - **Model export**
   - **ONNX** (`model.export_onnx(...)`) — run your model anywhere ONNX Runtime works
   - **marquetry archive** (`model.export_archive(...)`, `.mq`) — graph + weights in one
     file; reload it without the original model class, and keep training the restored model
   - `save_params` / `load_params` — plain npz weight checkpoints
 - **GPU support** via CuPy (optional)

## Quick Start
```python
import numpy as np

import marquetry
import marquetry.functions as funcs
from marquetry import optimizers, transformers
from marquetry.dataloaders import DataLoader
from marquetry.datasets import Spiral
from marquetry.models import MLP

train_set = Spiral(transform=transformers.ToFloat(), target_transform=np.argmax)
train_loader = DataLoader(train_set, batch_size=30)

model = MLP([32, 32, 3], activation=funcs.relu, is_dropout=False)
optimizer = optimizers.Adam().prepare(model)

for epoch in range(100):
    for x, t in train_loader:
        loss = funcs.softmax_cross_entropy(model(x), t)
        model.clear_grads()
        loss.backward()
        optimizer.update()

model.export_onnx(x, "spiral.onnx")     # needs the `onnx` package
model.export_archive(x, "spiral.mq")    # marquetry archive: graph + weights
```

More runnable examples live in [`samples/`](samples/README.md) — autograd basics,
FashionMNIST training, LSTM time-series forecasting, classic ML, and model export.

# Installation
```shell
pip install marquetry
```

With the ONNX export feature:
```shell
pip install "marquetry[onnx]"
```

### Dependencies
 - [Python](https://docs.python.org/3/) 3.10 or later
 - [NumPy](https://numpy.org/) >= 2.0
 - [pandas](https://pandas.pydata.org/) >= 2.2
 - [Pillow](https://pillow.readthedocs.io/en/stable/) >= 10.4
 - [SciPy](https://scipy.org/) >= 1.13

Optional
 - [onnx](https://onnx.ai/) — ONNX export (`pip install "marquetry[onnx]"`),
   and [onnxruntime](https://onnxruntime.ai/) to run the exported models
 - [CuPy](https://cupy.dev/) — GPU support
 - [Graphviz](https://graphviz.org/) — computation graph visualization (`Model.plot`)

For running the test suite
 - [pytest](https://docs.pytest.org/), [PyTorch](https://pytorch.org/),
   [scikit-learn](https://scikit-learn.org/stable/), [matplotlib](https://matplotlib.org/),
   onnx and onnxruntime

The test suite runs on Python 3.10 through 3.14 in CI.

### License
This project is licensed under the [MIT License](LICENSE.md).
