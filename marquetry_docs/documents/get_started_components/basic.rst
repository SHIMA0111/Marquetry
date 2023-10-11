Basic of Deep Learning
=======================

Let's Learn the deep learning!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this section, we learn the details of the deep learning!
Currently, we often hear "Deep Learning" but most people unknown the details.

Deep Learning(Neural Network) is a little complex but very exciting!

Let's dive into the wonderful deep learning world!


What is the "Learning"?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Recently, we heard or seen "Learning", Machine Learning, or Deep Learning or Reinforcement Learning, or so.
However, what is the meaning of the "Learning" originally?

A pioneer of Artificial Intelligence(AI), ``Arthur Lee Samuel`` who is a computer scientist
and held positions such as Standford University, he defines Machine Learning as

   Field of study that gives computers the ability to learn without being explicitly programmed

This is the most simple but precise definition.

In other words, what should we do is consider how to teach the relation of data and correct value to computer
without any formula directly relating to the problem.

To tell you the truth, there are a variety of approaches, tree model using "Impurity formula and Info Gain" indicator,
Support Vector Machine(SVM) using a distance from the support vector(record of the two-value which should be classified),
and so.

Sorry, I don't explain these algorithms here, these are more complex than the Gradient Descent method used by
Linear regression, Logistic regression, and almost all Neural Network(Deep Learning) algorithms.

Then, what do you think we learn about here?
... Oh, yes! We learn about the ``Gradient Descent method``! You know it well!
Hum? Oh, I said it myself earlier! lol

.. centered:: Let's learn the Gradient Descent method!!

Gradient Descent
~~~~~~~~~~~~~~~~~
``Gradient Descent`` is one of the most used algorithms.

``Linear Regression`` and ``Logistic Regression`` and ``Neural Network(Deep Learning)``, and so are following this
method.

This method's core is ``Gradient`` of ``Loss`` with respect to each parameter in the model.

``Gradient`` means partial differentiation so this method uses the slope of the loss with respect to each independent
parameters.

Why...?
********

:You: Why do we use partial differentiation to learn the computer from the data?
:Me:
   Let's come back to the what is the ``Loss``.
   ``Loss`` means the distance between the machine output and the correct value.

   Originally, Machine Learning's goal is that predict unknown things by provided data.
   If the prediction output is quite different from the correct value, the model is far from the actual event.

   For example, the data is ``Height`` and the correct value is ``Weight``, the correct dataset is the below.

   ======= =======
   Height  Weight
   168     55
   172     64
   180     75
   ======= =======

   However, the model predict the below

   ======= =======
   Height  Weight
   168     39
   172     73
   180     42
   ======= =======

   Then, this model doesn't fit the real world. The correct value can be rephrased as the ideal model.
   Intuitively, you can understand this situation means that the model is far from the ideal model.

   To predict correctly, we teach how the distance can be reduced to the ideal model.

   From the previous explanations, the distance is the same as ``Loss`` so to reduce the distance,
   we should teach only how to reduce the ``Loss``.

   Then, please try to remember, that the differentiation calculates the slope of the tangent at the point of function.
   The slope indicates the direction of the increasing function result so that the opposition indicates
   the decreasing direction.

   You can already see it! What should we do is only
   computing ``Gradient`` of the model including the output loss calculator and
   updating the model based on the ``Gradient``.

   Almost ``Machine Learning`` is structured by amount many parameters so the partial differentiation is
   very helpful computing the ``Gradient`` for each parameter.

.. tip::
   ``Loss`` is often called as ``Error``.

Gradient Descent subtracts the ``Gradient`` from the current params.
From the explanation, the ``Loss`` is decreased by this process so that the model is fit to the correct value.

Loss is ``Descended`` by ``Gradient`` so this method is called ``Gradient Descent method``.

Well, how to define the ``Loss`` by the output and the correct value.


Loss Function
~~~~~~~~~~~~~~
In the previous step, we learned the method of the fitting model to the correct value.

The method uses the ``Gradient`` of the output value's loss.
But we should how to calculate the loss value. The loss value calculator is called ``Loss Function``.

This function is independent of the model.
Loss function receives the prediction(output) and the correct value and calculates the loss value.

There are variable loss functions but here, I'd like to introduce 2 loss functions which commonly used.

The first one is for the Regression problem.

This loss function is called ``Mean Squared Error`` a.k.a ``MSE``.
The function's formula is

   MSE = Σ{(predict_value - correct_value)^2}

This is very simple, sum all squared residual values of each record.
Why is squared? Because the raw residual has mixed +/- value if just sum the value,
the loss is incorrect by the loss value annihilation.
So the sign of the residual is ignored together by squared.


The next one is for the Classification problem.

This loss function is called ``Cross Entropy Error``.
The formula is a little more complex than the ``MSE`` like the one below

   For one-hot data:
      CE = -Σ{correct_label * log(predict_value)}

.. tip::
   The one-hot formula is famous as ``Cross Entropy Error``. In this case, the ``correct_label`` is 0 or 1.
   Therefore, the actual mean of the formula is the sum all of the logarithm of
   the prediction score corresponding correct label.

   Other value is ignored because it is erased by the correct_label being 0.

For label encoded correct data, there is the more efficient formula.

   For label data:
      CE = -Σ{log(predict_matrix[index, correct_label])}

This function uses the logarithm specification.
The logarithm output is shapely decreasing when the value is closing to 0 so that the minus logarithm is
shapely increasing.

.. grid:: 2

   .. grid-item::
      .. image:: ../../_static/img/log_func.png

   .. grid-item::
      .. image:: ../../_static/img/log_func2.png

Please try to remember, ``Cross Entropy``'s core is the logarithm of the predicted score **corresponding correct label**.
The optimal loss function should be a large value when the prediction score is low.

This requirement is filled by the minus logarithm function.
So the ``Cross Entropy`` is structured by the -log function.
Another parameter is for fitting the loss function(Ignore the score corresponding wrong label).

.. caution::
   ``Cross Entropy Error`` is expected that the predicted scores are in the range from 0.0 to 1.0.
   The classification model output is generally a possibility. So, the value isn't over 1.0 and under 0.0.
   However, there is a case that the output doesn't settle into the range when you use multiple classification models.
   In the multiple predictions, the prediction value is converted to the range by :class:`marquetry.functions.softmax`
   but in the prediction phase, almost the framework doesn't use the Softmax function for reducing the computation cost.
   If you mistake the model setting, this function can return an error. Please caution such case.

Backpropagation algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, the weapons of ``Gradient Descent`` are in your hand!

The last piece connecting the weapons is ``Backpropagation``.

Before telling you the details, please consider how to find the gradient of the model and the loss function.
The loss function and the model are independent without the point that the loss function receiving
the model output of each other. So the gradient can't be found simply.

.. tip::
   To tell the truth, such connecting different functions are called ``composite functions``.
   In this function, the one function's output is the after function's input.

In this situation, we use ``Chain Rule`` of the differentiation.
This is a key man connecting the gradient of the independent functions.

Do you know the differentiation chain rule?

...Oh, no problem! I explain this here!

Let's consider the below case

There are two functions
   - f(x)
   - g(h)

And these functions' relation is the below
   - h = f(x)
   - y = g(h)

So this can be considered as composite
   - y = g(f(x))

When you find the gradient of ``y`` with respect to x by this function,
the required value can be expressed by the below

   dy/dx = (g(f(x)))'

But the value is calculated directly seems to be difficult...

``Chain Rule`` is where we come in!

From the composition definition, there is ``h`` as intermediate data.
Using this value, the ``dy/dx`` can be separated into the gradient of ``h`` with respect to ``x`` and
gradient ``y`` with respect to ``h``.

   dh/dx = f'(x)
   dy/dh = g'(h)

Then, we can express the ``dy/dx`` as

   dy/dx = dy/dh * dh/dx = f'(x) * g'(x)

In short, the composite functions of different functions can be differentiated by
the product of each function's differentiations.

This is ``Chain Rule``!

As a matter of fact, this ``Chain Rule`` can apply to 3 or more depth composite functions too.
This is very helpful to Neural Network(Deep Learning).

Following this rule, you can differentiate any complicated functions.

   h = sigmoid(x)
   y = h ** 2

   dh/dx = sigmoid(x) * (1 - sigmoid(x))
   dy/dh = 2 * h = 2 * sigmoid(x)

   dy/dx = (dy/dh) * (dh/dx) = {2 * sigmoid(x)} * {sigmoid(x) * (1 - sigmoid(x))}
         = 2 * (sigmoid(x)) ** 2 * (1 - sigmoid(x)) = 2 * h ** 2 * (1 - h)

Let's check the correctness!
The x is 4.

.. code-block:: python

   import marquetry as mq

   x = mq.array(4)

   h = mq.functions.sigmoid(x)
   y = h ** 2

   print(y)

The output is
   container(0.9643510838246173)

From the above formula, calculate the differentiation. And the comparison value is calculated by :meth:`backward`.

.. code-block:: python

   grad_x = 2 * (h ** 2) * (1 - h)

   y.backward()
   comparison_grad_x = x.grad

   if grad_x == comparison_grad_x:
       print("Your formula for differentiation is correct!!")

The output is
   Your formula for differentiation is correct!!

You were able to calculate the complicated function(:math:`sigmoid(x)^2`) by your hand!

What do you think? Do you understand how useful the ``Chain Rule``?

Actually, ``Chain Rule`` positions the recent Deep Learning's core. Furthermore,
:mod:`Marquetry` is also constructed based on this rule.

Now you get ``Gradient Descent`` and ``Backpropagation``! All you need is in your hands!

After here, we confirm how is the backpropagation used for Deep Learning,
Activation Function which is important for deep learning, and the last,
we try to construct a simple model by hand without any framework!

.. centered:: So let's continue to have fun as we go along!

Deep Learning backpropagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Well, we talk focusing about on Deep Learning!

At first, we need to understand ``Linear Regression`` mechanism.
Originally, ``Linear Regression`` is based on ``Linear Transformation (Linear Mapping)``.

More simply, the transformation means the below formula
   :math:`y = x_1w_1 + x_2w_2 + ... + x_nw_n + b`
      - X(input): (x\ :sub:`1`\ , x\ :sub:`2`\ , ..., x\ :sub:`n`\ )
      - W(weight): (w\ :sub:`1`\ , w\ :sub:`2`\ , ..., w\ :sub:`n`\ )
      - b(bias)

The `y` is Linear Regression output.

.. tip::
   In the ``Logistic Regression``, `y` is the input of the Logistic Sigmoid(:class:`marquetry.functions.sigmoid`).
   The sigmoid function output is the prediction score(this value settles into 0.0 ~ 1.0 by the sigmoid function).

   Therefore, ``Logistic Regression`` is the combination of ``Linear Regression`` and ``Logistic Sigmoid``.

Actually, the neurons in deep learning are processing this ``Linear Transformation`` and one new function.

The part of the new function is called ``Activation Function``.
Activation is profound so I'll explain the activation details later...
In this time, please keep your mind only about ``Activation receives the Linear Transformation outputs as inputs``.

The outputs is treated as inputs, we've seen the relation before!
Yes, this is a composite function.

The one neuron outputs only 1 value. In short, the layer has 2 neurons and the input has 3 dims, the value is transformed
as

:math:`y_1 = x_1w_{11} + x_2w_{12} + x_3w_{13} + b_1`

:math:`y_2 = x_1w_{21} + x_2w_{22} + x_3w_{23} + b_2`

- X(input): (x\ :sub:`1`\ , x\ :sub:`2`\ , x\ :sub:`3`\ )
- W\ :sub:`1`\ (neuron1_weight): (w\ :sub:`11`\ , w\ :sub:`12`\ , w\ :sub:`13`\ )
- W\ :sub:`2`\ (neuron2_weight): (w\ :sub:`21`\ , w\ :sub:`22`\ , w\ :sub:`23`\ )
- b\ :sub:`1`\ (neuron1_bias)
- b\ :sub:`2`\ (neuron2_bias)

From these, the output is (y\ :sub:`1`\ , y\ :sub:`2`\ )
so the number of outputs matches the number of neurons.
If you set layer after this layer, these outputs are treated as inputs of the next layer.

From a macro perspective, even the layer can be also considered as one function.
The relation between the layer and the next layer is also a composite function.

From there, You already know that deep learning is a big composite function built by simple small functions.

And these are composite functions, you know, the deep learning's gradient can be computed by
the ``Composite function's differentiation Chain Rule``.

Here's where this rule helps deep learning!!
By this rule, we can apply ``Gradient Descent`` to the Deep Learning model!

In Deep Learning, the function is very long so the differentiation chains as very long.
This method looks to propagate the ``Loss`` backward direction.

Therefore, this method is called an ``Error Backpropagation method`` especially.
And updating the model's parameters by the gradients as same as the Linear Regression and so on.

.. tip::
   Generally, the Linear Transformation uses matrix operation.
      - X (input_data): matrix (batch_size * data_dims)
      - W (weights): matrix (data_dims * neuron_nums)
      - b (bias): vector (neuron_nums)

   .. centered::
      :math:`Y = X･W + b`

   .. math::
      X =
      \underbrace{ \left.
      \begin{pmatrix}
      x_{11} & x_{12} & \cdots & x_{1m} \\
      x_{21} & x_{22} & \cdots & x_{2m} \\
      \vdots & \vdots & \ddots & \vdots \\
      x_{l1} & x_{l2} & \cdots & x_{lm} \\
      \end{pmatrix}
      \right\}}_{\text{$m$columns}}
      \,\text{$l$rows},
      W =
      \underbrace{ \left.
      \begin{pmatrix}
      w_{11} & w_{12} & \cdots & w_{1n} \\
      w_{21} & w_{22} & \cdots & w_{2n} \\
      \vdots & \vdots & \ddots & \vdots \\
      w_{m1} & w_{m2} & \cdots & w_{mn}
      \end{pmatrix}
      \right\}}_{\text{$n$columns}}
      \,\text{$m$rows},
      b =
      \underbrace { \left.
      \begin{pmatrix}
      b_1 & b_2 & \cdots & b_n
      \end{pmatrix}
      \right\}}_{\text{$n$columns}}

   The output `Y` is

   .. math::
      Y =
      \underbrace{ \left.
      \begin{pmatrix}
      \Sigma{(x_{1p}w_{p1}) + b_1} & \Sigma{(x_{1p}w_{p2}) + b_2} & \cdots & \Sigma{(x_{1p}w_{pn}) + b_n} \\
      \Sigma{(x_{2p}w_{p1}) + b_1} & \Sigma{(x_{2p}w_{p2}) + b_2} & \cdots & \Sigma{(x_{2p}w_{pn}) + b_n} \\
      \vdots & \vdots & \ddots & \ vdots \\
      \Sigma{(x_{lp}w_{p1}) + b_1} & \Sigma{(x_{lp}w_{p2}) + b_2} & \cdots & \Sigma{(x_{lp}w_{pn}) + b_n}
      \end{pmatrix}
      \right\}}_{\text{$n$columns}}
      \,\text{$l$rows}

   The matrix operation can be realized by NumPy. Marquetry also realize it but the internal flow uses NumPy.

   .. code-block:: python

      import numpy as np

      x = np.random.randn(3, 4)
      w = np.random.randn(4, 2)
      b = np.zeros(2)

      y = x.dot(w) + b

      print(y.shape)
      # (3, 2)

Activation
~~~~~~~~~~~
Here, we learn the Activation function which is a part of neuron components.

This function is very important for deep learning.

Why...?
********

:You: Why is the activation very important for deep learning?
      I think the most important thing about deep learning is piling up the layers.
:Me:
   Certainly, yes. But if the deep learning doesn't have the activation function, the deep learning can't
   get such rich expressive power even if the layer is piled up many.
   Let's test the deep learning with/without activation!

   The first one is using activation called ``ReLU``.

   .. code-block:: python

      import matplotlib.pyplot as plt
      import marquetry as mq

      dataset = mq.datasets.Spiral()
      dataloader = mq.dataloaders.DataLoader(dataset, batch_size=32)

      model = mq.models.MLP([128, 32, 3], activation=mq.functions.relu, is_dropout=False)
      optim = mq.optimizers.Adam().prepare(model)

      total_epoch = 1000
      interval = 100

      for epoch in range(total_epoch):
          x0, y0 = [], []
          x1, y1 = [], []
          x2, y2 = [], []

          total_loss = 0
          iterations = 0

          for x, t in dataloader:
              iterations += 1

              y = model(x)
              t = t.argmax(axis=1)
              loss = mq.functions.classification_cross_entropy(y, t)

              model.clear_grads()
              loss.backward()
              optim.update()

              pred = y.data.argmax(axis=1)
              total_loss += float(loss.data)

              for i, predict_num in enumerate(pred):
                  if predict_num == 0:
                      x0.append(x[i, 0])
                      y0.append(x[i, 1])
                  elif predict_num == 1:
                      x1.append(x[i, 0])
                      y1.append(x[i, 1])
                  else:
                      x2.append(x[i, 0])
                      y2.append(x[i, 1])

          if epoch % interval == 0:
              plt.scatter(x0, y0)
              plt.scatter(x1, y1)
              plt.scatter(x2, y2)
              plt.title("{} epoch, loss: {:.4f}".format(epoch, total_loss / iterations))
              plt.show()

          print("Epoch: {} / {}, Loss: {:.4f}".format(epoch, total_epoch, total_loss / iterations))

   .. grid:: 2
      :gutter: 2

      .. grid-item::
         .. image:: ../../_static/img/spiral_0.png

      .. grid-item::
         .. image:: ../../_static/img/spiral_900.png

   This model learning correctly. Then, we use the same setting except for ``activation``.
   We use :meth:`marquetry.functions.identity` which returns the input data unchanged
   (so this is no activation substantially).

      .. code-block:: python

         ...
         # change only model definition.
         model = mq.models.MLP([128, 32, 3], activation=mq.functions.identity, is_dropout=False)
         ...

   .. grid:: 3
      :gutter: 2

      .. grid-item::
         .. image:: ../../_static/img/spiral_non_act_0.png

      .. grid-item::
         .. image:: ../../_static/img/spiral_non_act_200.png

      .. grid-item::
         .. image:: ../../_static/img/spiral_non_act_400.png

      .. grid-item::
         .. image:: ../../_static/img/spiral_non_act_600.png

      .. grid-item::
         .. image:: ../../_static/img/spiral_non_act_800.png

      .. grid-item::
         .. image:: ../../_static/img/spiral_non_act_900.png

   This model can't learn this data correctly. In the model without activation, the model expression isn't enough...

   To tell you the truth, the ``Activation`` function governs non-linear expression.
   If the activation function isn't used, the model even if deep can't express non-linear features.

   What do you think? Can you understand why ``Activation`` is very important? hahaha!

This phenomenon can be explained by the transformation result simply (I describe the actual flow in the ``note`` section).
Anyway, in short, if the activation function isn't used, the model computation can be converted simple linear formula even
how the neuron and layer are large.

.. note::
   Check Non-activation deep learning model calculation!
   The parameter is below. (To be simple, the biases are abbreviated.)

   .. math::
      X =
      \begin{pmatrix}
      x_1 & x_2 & x_3
      \end{pmatrix},
      W_1 =
      \begin{pmatrix}
      w^1_{11} & w^1_{12} & w^1_{13} \\
      w^1_{21} & w^1_{22} & w^1_{23} \\
      w^1_{31} & w^1_{32} & w^1_{33}
      \end{pmatrix},
      W_2 =
      \begin{pmatrix}
      w^2_{11} \\
      w^2_{21} \\
      w^2_{31}
      \end{pmatrix}

   The actual calculation is the below.

   .. math::
      Y_1 = X･W_1 =
      \begin{pmatrix}
      \Sigma(x_pw^1_{p1}) & \Sigma(x_pw^1_{p2}) & \Sigma(x_pw^1_{p3})
      \end{pmatrix}

      Y_2 = Y_1･W_2 =
      \begin{pmatrix}
      \Sigma\{\Sigma(x_pw^1_{pq})w^2_{q1}\}
      \end{pmatrix}

   Expanding the matrix, the Y\ :sub:`2`\ can convert the below

   .. math::
      Y_2 = (w^1_{11}w^2_{11} + w^1_{12}w^2_{21} + w^1_{13}w^2_{31}) * x_1 + (w^1_{21}w^2_{11} + w^1_{22}w^2_{21} +
      w^1_{23}w^2_{31}) * x_2 + (w^1_{31}w^2_{11} + w^1_{32}w^2_{21} + w^1_{33}w^2_{31}) * x_3

   The weight is all float values so :math:`(w^1_{11}w^2_{11} + w^1_{12}w^2_{21} + w^1_{13}w^2_{31})` and so are
   also simple float value.
   Therefore, this transformation can be considered as Linear Regression using the below weight.

   .. math::
      W_{total} =
      \begin{pmatrix}
      w^1_{11}w^2_{11} + w^1_{12}w^2_{21} + w^1_{13}w^2_{31} \\
      w^1_{21}w^2_{11} + w^1_{22}w^2_{21} + w^1_{23}w^2_{31} \\
      w^1_{31}w^2_{11} + w^1_{32}w^2_{21} + w^1_{33}w^2_{31}
      \end{pmatrix}

   Please remember the Linear Regression. This is simple Linear Transformation so the output is also linear separation.
   This trait is unchanged even if there are tens of millions layers.

A Deep Learning model without activation function can't express non-linear separation but applying activation to the model,
expression of the model is rapidly increasing.
Let me rephrase this, ``Activation`` is one of the deep learning mechanism core.

For non-linear transformation, the activation function needs to be a non-linear function, of course.

Currently, :meth:`marquetry.functions.relu` is usually used as an activation function so the next step scratches deep learning
using this activation.

.. tip::
   Of course, there are a variety of activations so you need to consider and choose an activation function to match
   your use case.

   However, in many cased, you can use ``ReLU`` function. This function is very simple so the computation cost is low
   and match many cases.
   If you are NOT a specialist in the use case area, please try to use ReLU at first!


.. centered:: Thank you for your hard work by here!

Now we have gotten all the weapons to scratch deep learning!
Let's start the final quest of this section!!

Scratch Deep Learning
~~~~~~~~~~~~~~~~~~~~~~
Finally, you challenge this largest quest!
We learned Machine Learning and Deep Learning here. Now, we get all needed for Deep Learning construction!

.. centered:: Let's show the culmination of our effort!

Some of the order of the implementation is different from the explanation to easily understand the flow.

.. note:: We use :mod:`NumPy` to calculate the matrix operation.

1. Implement ReLU(Rectified Linear Unit)

   The formula is

   .. math::
      y = \{x \ \textrm{(if x >= 0)}, 0 \ \textrm{(if x < 0)}\}

   .. image:: ../../_static/img/relu_fig.png
      :align: center

   .. code-block:: python

      import numpy as np

      class ReLU(object):
          def __init__(self):
              self.mask = None

          def forward(self, x):
              self.mask = np.asarray(x >= 0, dtype="f")

              y = np.where(x < 0, 0, x)
              return y

          def backward(self, grad_y):
              grad_x = grad_y * self.mask

              return grad_x


   In the backward method, we use ``mask`` which depends on the forward input.
   This mask is mapped 0 as x(input) < 0, otherwise, 1.

   The ``backward`` returns the output differentiation with respect to `x`.

2. Implement Linear trans function

   The formula is

   .. math::
      y = x\ @\ w + b\ \textrm{(@ means dot product)}

   .. code-block:: python

      class Linear(object):
          def __init__(self, n_neuron, input_size=None, init_std=0.01):
              self.n_neuron = n_neuron
              self.input_size = input_size
              self.init_std = init_std

              self.w = None
              if self.input_size is not None:
                  self.w = np.random.randn(input_size, n_neuron) * init_std

              self.b = np.zeros(n_neuron)

              self.x = None
              self.grad_w = None
              self.grad_b = None

          def forward(self, x):
              if self.input_size is None:
                  self.input_size = x.shape[1]

                  self.w = np.random.randn(self.input_size, self.n_neuron) * self.init_std

              self.x = x
              y = np.dot(x, self.w) + self.b

              return y

          def backward(self, grad_y):
              grad_b = grad_y.sum(axis=0)
              grad_w = np.dot(self.x.T, grad_y)
              grad_x = np.dot(grad_y, self.w.T)

              self.grad_w = grad_w
              self.grad_b = grad_b

              return grad_x

          def update(self, learn_rate=0.01):
              if self.w is None:
                  raise Exception("Please do backward first!")

              self.w -= learn_rate * self.grad_w
              self.b -= learn_rate * self.grad_b

   ``init_std`` controls the initial weight standard deviation.
   Default is 0.01.

   ``update`` method updates the layer parameters.

   Do you remember what the optimizer's name is? This method updates the parameters based on only the latest gradient.

   ... Yes, this is SGD method!

3. Implement SoftmaxWithCrossEntropy which loss function for classification problems.

   The formula is

   .. math::
      Softmax = exp(x) / \Sigma exp(x_k)

      CrossEntropy = -\Sigma \{t_k * log(x_k)\}

   .. code-block:: python

      class SoftmaxWithCrossEntropy(object):
          def __init__(self):
              self.softmax_data = None
              self.t = None

          def forward(self, x, t):
              if x.ndim == 1:
                  x = x.reshape(1, x.size)
                  t = t.reshape(1, t.size)

              x_clip = x - x.max(axis=1, keepdims=True)
              softmax_data = np.exp(x_clip) / np.sum(np.exp(x_clip), axis=1, keepdims=True)

              if x.size == t.size:
                  t = t.argmax(axis=1)

              if t.ndim == 2:
                  t = t.flatten()

              self.t = t
              self.softmax_data = softmax_data

              batch_size = softmax_data.shape[0]

              loss = -np.sum(np.log(softmax_data[np.arange(batch_size), t] + 1e-8)) / batch_size

              return loss

          def backward(self, grad_y=1.):
              batch_size = self.t.shape[0]

              grad_x = self.softmax_data

              grad_x[np.arange(batch_size), self.t] -= 1.
              grad_x *= grad_y
              grad_x = grad_x / batch_size

              return grad_x

   The ``x_clip`` is to prevent overflow. :math:`exp(x)` means :math:`e^x` so if the x is over 710,
   the result is ``inf``.

   In such a case, the computation can't be continued. In this class,
   subtracting the row max from all the row values deal with this issue.

   .. tip::
      Why can we subtract the max from values? Can it change the result?

      Perhaps, you have such a question for this operation, but this operation can't change the outputs.
      Let's check this!

      Originally, the ``softmax`` is a monotonically increasing function.

      .. image:: ../../_static/img/softmax.png
         :align: center

      A monotonically increasing function does the scaling but not change the values relationship.
      Please remember, that the prediction score only depends on the magnitude relationship
      and softmax is monotonically increasing, so softmax is just scaling(value to relative probability),
      not changing the magnitude relations.
      The subtracting max value is also only scaling, not changing the magnitude relations
      so the final output isn't affected by this operation.

   And the ``1e-8`` in the loss variable is also to prevent overflow.
   If the log(x) receives 0, it returns ``-inf``.

   .. tip::
      The ``SoftmaxWithCrossEntropy``'s backward can be calculated by

      .. math::

         f(x) = exp(x) / \Sigma exp(x_l) \\
         g(f(x_k)) = -\Sigma \{t_k * log(f(x_k))\}

      So the composite function can be considered as

      .. math::

         g(f(x_k)) = -\sum_{k=0} \{t_k * log(exp(x_k) / \sum_{l=0} exp(x_l))\}

      Expanding this function like below

      .. math::
         \begin{align}
         g(f(x_k)) &= -\sum_{k=0} \{t_k * log(exp(x_k) / \sum_{l=0} exp(x_l))\} \\
         &= -\sum_{k=0} \{t_k * (log(exp(x_k)) - log(\sum_{l=0} exp(x_l)))\} \\
         &= -\sum_{k=0} \{t_k * log(exp(x_k)) - t_k * log(\sum_{l=0} exp(x_l))\} \\
         &= -\sum_{k=0} \{t_k * x_k - t_k * log(\sum_{l=0} exp(x_l))\} \\
         &= \sum_{k=0} \{t_k * log(\sum_{l=0} exp(x_l))\} - \sum_{k=0} \{t_k * x_k\} \\
         &= (\sum_{k=0} t_k) * log(\sum_{l=0} exp(x_l)) - \sum_{k=0} (t_k * x_k) \\
         &\text{$t_k$ is one-hot data so the $\sum_{k=0}  t_k$ is 1.} \\
         &= log(\Sigma exp(x_l)) - \sum_{k=0} (t_k * x_k) \\
         \end{align}

      The gradient of :math:`log(x)` is :math:`1/x` and the gradient of :math:`\sum_{l=0} exp(x_l)` is :math:`exp(x_k)`,
      so the gradient of :math:`log(\sum_{l=0} exp(x_l))` is :math:`exp(x_k)/\sum_{l=0} exp(x_l)`.

      And the :math:`\sum_{k=0} (t_k * x_k)` means :math:`(t_1*x_1 + t_2*x_2 + ... + t_k*x_k + ... + t_n*x_n)`.
      The gradient is :math:`t_k` (This partial differentiation respects to :math:`x_k`
      so :math:`t_1*x_1`, :math:`t_2*x_2`, and so are constant value (deleted by the partial differentiation).)

      Therefore, the gradient is

      .. math::
         \begin{align}
         g'(f(x_k)) &= \{exp(x_k) / \sum_{l=0} exp(x_l)\} - t_k \\
         &= f(x) - t_k
         \end{align}

4. Create a model object using implemented classes

   Now, we have all the components for deep learning construction but difficult to use it is now.
   Therefore, we create wrapping objects to use easily.

   .. code-block:: python

      class SimpleModel(object):
          def __init__(self, output_size, middle_neuron=32, input_size=None):
              self.layers = [
                  Linear(middle_neuron, input_size),
                  ReLU(),
                  Linear(output_size)
              ]
              self.loss_func = SoftmaxWithCrossEntropy()

              self.loss = 0.
              self.iterations = 0

          def predict(self, x):
              for layer in self.layers:
                  x = layer.forward(x)

              return x

          def fit(self, x, t, max_epoch=1000, batch_size=32, data_shuffle=True, intervals=50):
              data_size = len(t)
              max_iterations = data_size // batch_size

              if max_iterations == 0:
                  raise Exception("batch_size is {} so the input data size needs over than the batch_size but got {}-records"
                                  .format(batch_size, data_size))

              self.iterations = 0

              for epoch_num in range(max_epoch):
                  if data_shuffle:
                      random_index = np.random.permutation(data_size)
                  else:
                      random_index = np.arange(data_size)

                  for iterations in range(max_iterations):
                      self.iterations += 1

                      batch_random_index = random_index[iterations * batch_size:(iterations + 1) * batch_size]

                      batch_x = x[batch_random_index]
                      batch_t = t[batch_random_index]

                      out = self.predict(batch_x)

                      loss = self.loss_func.forward(out, batch_t)

                      grad_x = self.loss_func.backward()
                      for layer in reversed(self.layers):
                          grad_x = layer.backward(grad_x)

                      for layer in self.layers:
                          layer.update()

                      self.loss += loss

                      if self.iterations % intervals == 0:
                          print("The loss is {:.4f}".format(self.loss / intervals))

                          self.loss = 0.


   This class wraps the operation of the layer. Using this, the user should do only input the data and the label.

   .. tip::
      If you create some CLI app for Training model, it is kind for a user that
      display the proceed like loss value or accuracy or so.
      In this implementation, we display only the loss value,
      but if the epochs or iterations numbers are displayed in the output text, it is very helpful to a user!

      Let's try to modify this code as such new implements!

5. Training the model

   Finally, you create deep learning by your hand!
   Now, let's let it learn!!
   In this training, we use the ``trigonometric area separation problem``
   which is created in the :doc:`./entrance`.

   .. code-block:: python

       x = np.arange(-1.0, 1.05, 0.05)
       y = np.arange(-1.0, 1.05, 0.05)

       data = []
       target = []

       for x_one in x:
           for y_one in y:
               data.append([x_one, y_one])

               if y_one < np.sin(x_one * np.pi):
                   target.append([0])
               else:
                   target.append([1])

       data = np.array(data)
       target = np.array(target)

       model = SimpleModel(2)

       model.fit(data, target)

       y = model.predict(data)

       x_0, y_0, x_1, y_1 = [], [], [], []

       for index, pred in enumerate(y):
           pred = pred.reshape(-1)

           if float(pred[0]) < float(pred[1]):
               x_0.append(data[index, 0])
               y_0.append(data[index, 1])

           else:
               x_1.append(data[index, 0])
               y_1.append(data[index, 1])

       plt.plot(x, np.sin(x * np.pi), linestyle="dashed")
       plt.scatter(x_0, y_0, marker="^")
       plt.scatter(x_1, y_1, marker="o", c="m")

       plt.show()

   Then, you can confirm the loss value is decreasing every displaying time.
   And last, you can see a beautiful area separation!

   .. image:: ../../_static/img/scratch_tri.png
      :align: center

   .. tip::
      The model we created here is specialized to the classification problem only, not the regression problem.
      However, we can use this for various classification problems.

      Let's try and play various problems and modify the model!

Now, all program in this section is up! Thank you for your patience to the end!

In this section, we started a question  ``What is the "Learning"?``. And we've been through some mechanisms.
Finally, we created a neural network model by your hand without any Framework, and trained the model.

What do you think? If this paper helps your great journey!

Deep/Neural Network has a number of possibilities but the mechanisms are very simple.
So I think it is wasted that people mistakenly think it's too difficult and dislike it.
Probably, you didn't feel difficulty with each one components in this section.
However, such models(Deep/Neural Network) are still the front line of this world.
The world's top edge is also constructed by such simple function assembly.
*(Of course, there are many difficult field of research too...)*

Anyway, today you stepped out to one of the world's cutting-edge field!
I hope you keep trying something in this field.

Please don't forget, this is not a goal, you just stepped on the great journey.

----

I prepared the next step which tries a more practical problem using ``Marquetry``.
The first one is the ``Titanic Disaster`` prediction.

.. button-link:: ../trial_examples/titanic_disaster.html
   :color: info
   :outline:
   :expand:

   Titanic Disaster
