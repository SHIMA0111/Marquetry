Release Note
=============

Version 0.3.0 (Released: 2026/06/11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the final feature release of the pure-Python Marquetry.

:new features:

   - **ONNX export** (:meth:`marquetry.Model.export_onnx` / :func:`marquetry.onnx_export.export_onnx`)
      - Exports the traced inference graph (opset 21 / IR version 10 by default)
      - Dynamic batch axis support, validated against ONNX Runtime
      - Install the dependency with ``pip install "marquetry[onnx]"``
   - **Marquetry archive** ``.mq`` (:meth:`marquetry.Model.export_archive` / :func:`marquetry.model_archive.save_archive` / :func:`marquetry.model_archive.load_archive`)
      - Stores the computation graph and the weights in one file (zip of JSON + npz, no pickle)
      - The archive loads back without the original model class, and the restored
        model supports inference **and further training**
   - **SVM** (:class:`marquetry.ml.SVM`): linear support vector machine on the
     ``MachineLearning`` interface (soft margin by default, ``c=None`` for hard margin,
     accepts ``{0, 1}`` / ``{-1, 1}`` labels, save/load support)
   - **Batch Renormalization** (:class:`marquetry.functions.BatchRenormalization` /
     :class:`marquetry.layers.BatchRenormalization`, https://arxiv.org/abs/1702.03275 );
     ``rmax=1`` / ``dmax=0`` reduces exactly to batch normalization
   - Add runnable example scripts under ``samples/`` in the repository
   - Add ``Config.retain_graph_inputs`` which keeps all function inputs
     on the recorded graph (used by the exporters)

:changes:

   - **(Breaking)** :class:`marquetry.layers.BiLSTM` is reworked into a true bidirectional
     LSTM over ``(batch, time, features)`` sequences. The previous implementation reversed
     the feature axis, which was a no-op for the documented usage
   - **(Breaking)** :class:`marquetry.functions.MaxPooling2D` now pads with the lowest
     value instead of zero, following the standard max pooling semantics
     (PyTorch/Chainer/ONNX). Outputs change when ``pad > 0`` and a window contains
     only negative values; gradients no longer leak into padding cells
   - **(Breaking)** Supported Python versions are now 3.10 - 3.14 (was 3.8+)
   - :class:`marquetry.dataloaders.DataLoader` now returns the final partial batch
     (the PyTorch/Chainer default); ``SeqDataLoader`` keeps floor-based iteration
     for its parallel streams
   - Dependency floors are raised to the first NumPy-2-compatible line:
     ``numpy>=2.0`` / ``pandas>=2.2`` / ``Pillow>=10.4`` / ``scipy>=1.13``
   - Weight initialization now respects the declared ``dtype``:
     :class:`marquetry.layers.Linear`, :class:`marquetry.layers.Convolution2D` and
     :class:`marquetry.layers.Deconvolution2D` create float32 weights as documented
     (under NumPy 2 they were silently promoted to float64)
   - :class:`marquetry.layers.PReLU` and :class:`marquetry.layers.DynamicSwish`
     parameters default to float32 (``dtype`` argument added)
   - Hyperparameters are validated across components
     (``batch_size``, ``decay``, ``epoch``, ``learn_rate``, ``rmax``, ``dmax``, ...)
   - ``ColumnNormalize`` / ``ColumnStandardize`` preserve NaNs in zero-variance columns
     for downstream imputation
   - Remove the ``pre_implemetation`` directory (its drafts graduated to
     :mod:`marquetry.ml` and :mod:`marquetry.functions`)
   - The test suite grew from 318 to 613 tests and runs on Python 3.10 - 3.14 in CI

:bug fixes:

   **Autodiff engine**

   - ``Container.__gt__`` compared in the inverted direction
   - Non-inplace ``Container.astype`` returned a raw ndarray instead of a Container
   - ``Container.unchain_backward`` left the starting container chained and crashed
     on leaf containers
   - :meth:`marquetry.random_int` couldn't omit the ``high`` argument as documented
   - Python-scalar operands silently upcast float32 data to float64 under NumPy 2
     (weak-scalar promotion is now preserved)

   **Gradients**

   - ``MeanSquaredError`` backward multiplied the upstream gradient twice into one input
   - ``Dropout`` backward dropped the ``1/(1-rate)`` scale; eval-mode backward now passes
     gradients through
   - ``BatchNormalization`` 4D backward divided by ``N`` instead of ``N*H*W``;
     eval-mode backward used stale statistics
   - ``Conv2DGradW`` double-backward ignored the upstream gradient
   - ``softmax_cross_entropy`` backward built the one-hot matrix with the integer target dtype
   - Max pooling gradients routed into padding cells were silently dropped
     (fixed by the lowest-value padding change)
   - :class:`marquetry.functions.GELU` (``none`` / ``tanh``) upcast float32 inputs to float64

   **Optimizers / Layers**

   - ``MomentumSGD`` never used its learning rate
   - ``GRU`` skipped the update gate on the first step
   - ``Embedding.set_embedding_vector`` broke ``save_params`` while freezing the vectors

   **Preprocesses / ML / Misc**

   - ``MissImputation`` used the category statistic for numeric columns
   - ``ColumnStandardize`` swapped min/max (inverted scaling); zero-division guards
     added to normalize/standardize
   - ``LabelEncode`` / ``OneHotEncode`` treated subset categories at inference as unknown,
     and no longer emit pandas ``FutureWarning`` about ``replace`` downcasting on pandas 2.x
   - ``Compose()`` crashed with the default argument
   - ``RandomForest`` used class labels as column indexes and the same seed for every tree
   - Multi-class metrics rejected two-class logits; single-class evaluation batches are
     supported with proper zero-denominator handling
   - ``get_file`` failed on nested cache directories

Version 0.2.0 (Released: 2023/10/22)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:new features:

   - Add Layer Normalizer (:meth:`marquetry.functions.layer_normalization` and :class:`marquetry.layers.LayerNormalization`)
   - Add L2 Normalizer (:class:`marquetry.functions.l2_normalization`)
   - Add new activation functions
      - GELU (Gaussian Error Linear Units)
      - GLU (Gated Linear Units)
      - Mish (https://arxiv.org/abs/1908.08681 )
      - Swish (https://arxiv.org/abs/1710.05941 )
      - PReLU (Parametric Rectified Linear Units)
      - Softplus
   - Add new optimizer functions
      - AdaDelta
      - AdamW
      - AdaMax
      - Lion (EvoLved Sign Momentum https://arxiv.org/abs/2302.06675 )
      - NAdam (Nesterov Adoptive Moment Estimation)
      - Nesterov (Nesterov Accelerate Gradient Descent)
   - Add new mathmatics functions
      - Square (:class:`marquetry.functions.square`)
      - Sqrt (:class:`marquetry.functions.sqrt`)
   - Support custom csv file import (Beta)
      - CsvLoader (:class:`marquetry.datasets.CsvLoader`)
      - CustomDataset (:class:`marquetry.datasets.CustomDataset`)

:changes:
   - Batch Normalization support custom `eps` by the initializer
   - split function support section split and the argument is renamed as ``indices_or_sections``
   - Learning Rate argument's name is unified as ``lr`` for all optimizers

:bug fixes:
   - :meth:`marquetry.array` can't work when input `None`
   - :class:`marquetry.preprocesses.MissImputation` raise FutureWarning

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
   - :class:`Max` (and :class:`Min` which inherits the :class:`Max`) returns a strange form of values' array it can't compare with the native numpy ndarray

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
