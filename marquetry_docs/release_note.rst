Release Note
=============

Version 0.1.0 (Released: 2023/10/05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:new features:

   - Support directly set :class:`list`, :class:`int`, :class:`float` to container by :meth:`marquetry.array`
   - Support random generator using :meth:`marquetry.random`, :meth:`marquetry.random_int`, and :meth:`marquetry.random_gen`
   - Support tree model in :mod:`ml.tree`
   - Add :class:`Spiral` dataset
   - Support functions for helping the calculation
      - :meth:`absolute`
      - :meth:`tan`
      - :meth:`clip`
   - Add evaluation functions
      - For classification, ``accuracy, classification_error, f_score, precision, recall``
      - For regression, ``mean_absolute_error, mean_squared_error, r2_score, relative_absolute_error, relative_squared_error, root_mean_squared_error``
   - Add loss functions for regression
      - :meth:`mean_absolute_error`
   - Official docs

:changes:

   - Variable was renamed as :class:`Container`
   - Titanic can use without considering the past statistic file
   - Preprocess functions improve user experience (auto detect the changing prerequisite in the same dataset and so)
   - Refactoring the source code to more clear directory structure from the one sheet implementation
   - ML model and Preprocess functions inherits their base class (in 0.0.1, these don't have base class)

:bug fixes:

   - Normalize implementation wasn't correct
   - Memory usage reducing mechanism installing

Version 0.0.1 - 0.0.2 (Release: N/A (Test Release))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:new features:

   - Variable which is the base class holding the values
   - Functions which are basic for deep learning
   - Layers which are wrapping parameters of the layer
   - Built-in Datasets
      - MNIST
      - FashionMNIST
      - SinCurve
      - Titanic
   - CUDA support using :mod:`CuPy`
   - Models
      - Sequential which helps a user create own model
      - MLP (Multi Layer Perceptron)
      - CNN
   - Optimizers
      - SGD
      - MomentumSGD
      - AdaGrad
      - RMSProp
      - Adam

   And other of the Marquetry components. This is the first of the Marquetry.

:changes:
   N/A

:bug fixes:
   N/A
