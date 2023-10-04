Release Note
=============

Version 0.1.0 (Released: 2023/10/05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:new features:

   - Release Official docs
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
   - Add docstring for all functions and classes
   - Memory usage reducing mechanism called as "Aggressive Buffer Release" installing to the Container and the Function
   - Support user managed vector for :class:`Embedding` class
   - :class:`RNN` and :class:`LSTM`, and :class:`GRU` supports setting the hidden state via :meth:`set_state`
   - ML model and Preprocess functions inherits their base class (in 0.0.1, these don't have base class)

:changes:

   - Refactoring the source code to more clear directory structure from the one sheet implementation
   - The marquetry defined variable :class:`Variable` was renamed as :class:`Container`
   - Titanic can use without considering the past statistic file (delete the ``remove_old_static`` argument)
   - :class:`SinCurve` dataset was renamed to :class:`TrigonometricCurve`
   - :mod:`marquetry.preprocess` was renamed to :mod:`marquetry.preprocesses`
   - Preprocess functions improve user experience (auto detect the changing prerequisite in the same dataset and so)
   - Rename :meth:`logsumexp` to :meth:`log_sum_exp`
   - :meth:`repeat` support multi axis and ``repeat_num`` was renamed to ``repeats``
   - :class:`MatMul` and :meth:`matmul`'s arguments was renamed to `x0` and `x1` from `x1` and `x2`
   - :class:`Layer`'s save/load params method was renamed to :meth:`save_params` and :meth:`load_params`
   - Configurations are managed in :class:`Config`, and support the cache directory(default is ``~/.marquetry``) changing by user.
   - Change the module name for conventional_ml models from ``conventional_ml`` to ``ml``
   - ML model like :class:`RegressionTree`'s score method changed from the evaluator is only accuracy to user defined method.

:bug fixes:

   - Normalize implementation wasn't correct
   - :class:`UnSqueeze` doesn't work expectedly when the axis is specified multiple type (tuple or list)
   - :class:`Max` (and :class:`Min` which inherits the :class:`Max`) returns odd type values' array it can't compare with the native numpy ndarray

Initial Version (Not official release)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
