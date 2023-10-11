==================================
Entrance of Deep Learning
==================================

Let's start your journey!
--------------------------
Welcome to the Deep Learning World!!
Here is your start point!

What will do here?
~~~~~~~~~~~~~~~~~~~

:You: What will do here? We use a complex data?
:Me: No, here we use a simple and intuitive problem for training and testing.

Let's start learning!

Challenge!
-----------
We use simple separate area data.

What is the Dataset we use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The dataset is an area classification problem!
This problem's core is the grid data classifying if the grid is over the sin curve or under.
From our(people), the classification is simple and easy.
Because we can understand that this problem is simply as y > sin(x) or y < sin(x).

However, we don't provide this formula. We provide just the coordinate points
and the correct label corresponding to the points.

**Let's challenge the project with Marquetry!**


Create Data
~~~~~~~~~~~~
Let's create the dataset!
We prepare x coordinate which is the range [-1.0, 1.0] splitting each 0.05 and
y coordinate which is also the range [-1.0, 1.0] splitting each 0.05.
The number of coordinate points is (2.0 / 0.05) * (2.0 / 0.05) = 40 * 40 = 1600.
Therefore, the correct label is also 1600 records.

This time, we use :mod:`matplotlib` to visualize this data.

1. Import ``matplotlib``

   .. code-block::

      pip install matplotlib

The upper of the sin curve points are classified as 1, under points are 0.

2. Prepare the coordinate points

   .. code-block:: python

      import numpy as np

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

2. Let's check this data plotting the point separate the area using :mod:`matplotlib`

   .. code-block:: python

      import matplotlib.pyplot as plt


      for data_one, target_one in zip(data, target):
          if target_one[0] == 0:
              plt.plot(data_one[0], data_one[1], marker="^", c="m")
          else:
              plt.plot(data_one[0], data_one[1], marker="o", c="g")

      plt.show()

   Result example

   .. image:: ../../_static/img/area_data.png
      :align: center

You can see a beautiful sin curve!

To tell you the truth, this learning is difficult for conventional linear machine learning.
Because, conventional machine learning, which is ``linear regression`` or ``logistic regression``,
can learn only linear information.
However, the sin curve is a non-linear function so the classification threshold is also a non-linear.
In conventional machine learning, this learning is very hard.

I've tested the training using ``Logistic Regression``, please see the below figures.

The first figure is the unlearned model output. The second is the 100 epoch learned model.
The third figure is the 900 epoch learned model.

You can see the model can only have linear output. This doesn't fit the non-linear data(sin curve).

.. note::
   1 epoch means learning all dataset. In this time, the 1600 data is used.

   The details in :ref:`Epoch explanation <epoch>`.

.. grid:: 3
   :gutter: 2

   .. grid-item::

      .. image:: ../../_static/img/linear_sin_0.png

      Non Learned

   .. grid-item::

      .. image:: ../../_static/img/linear_sin_100.png

      100 epochs

   .. grid-item::

      .. image:: ../../_static/img/linear_sin_900.png

      900 epochs

.. tip::
   Keep in mind, there are non-linear models even in conventional machine learning models
   like ``polynomial regression`` and so.
   If you use such non-linear models, you can learn non-linear functions.

.. centered:: **Congratulation! You succeed the dataset creation!**

Create Model
~~~~~~~~~~~~~
Let's start model definition. Oh, rest assured!

Now you are using Marquetry, so the definition can be very easy.

This problem is **not** image data and not sequence data so we use a simple fully connected neural network.

A fully connected neural network means neurons in the current layer and
the next/previous layer are connected to each other.
Such neural network is sometimes called ``Multiple Layer Perceptron``.

Oh, sorry, I didn't explain what is the ``Neural Network``.

Neural Network is base of the Deep Learning. In other words, Deep Learning is a deeper Neural Network.
In some documents, ``Deep Learning`` and ``Neural Network`` are used indicating the same thing.

In typically, an upper than 3-layer neural network is often called ``Deep Learning``.
But the definition seems to be ambiguous.
You don't need to remember this! Please keep in the back of your mind only the Deep Learning is
deeper Neural Network so the mechanism is almost the same.

Well, this time, we create a 3-layer Neural Network. Using :class:`marquetry.models.MLP`.
   - What is the MLP? MLP stands for ``Multiple Layer Perceptron``!

1. Define the model, don't worry! You should do is only define the number of the neuron and the layer.

   .. code-block:: python

         import marquetry as mq
         model = mq.models.MLP([2, 3, 1], is_dropout=False)

   .. note:: The definition means the first layer has 2 neurons and　the second one has 3 neurons,
             and the last layer has 1 neuron.
             The last layer is called as ``output layer`` which must be the same size as the output that you want.
             In this time, the output is over/under so this can be expressed by 0/1 so the output size should **1**.

   .. tip:: In accuracy, Neural Network(Deep Learning) has one more layer which is called as ``input layer``.
            However, the input layer only forwards the input layer to the first layer.
            In other words, the input layer has no compute process.
            Therefore, the layer doesn't count as the model's layer in almost every case.

            However, some documents count layers including the input layer so if you face such documents,
            please remember this :)


In the network training, we need to compute the gradient for the loss of the output compared with the target data.
The loss is important to learn the excellence of the model for the time.

:You: What is the loss? Why is it needed?
:Me: Loss is the distance for the ideal!
     Please imagine when you studied something, maybe you tried and made mistakes.
     From mistakes, you can learn how to make no mistakes after this time.
     Neural Network is inspired by human cranial nerves so to make the model learn by itself,
     we need to provide the correct error(mistakes) as we did.
     The correct error in the neural network is called ``loss`` and it is provided by ``Loss Function``.

.. tip::
    Let me rephrase it, the correct data is the ideal output. If the loss(error) is 0,
    the model can provide a completely correct prediction.
    In this situation, the model has nothing to learn from the data.

    However, if the loss(error) is larger than 0, the model can learn the relation between the data and the ideal output.
    In general, Neural Network learn data to reduce the loss(error) by updating the model parameter.
    And when getting closer to the ideal, the model loss is also reduced.

    Therefore, ``Loss`` can be told about as the distance for the ideal.

2. Define ``Loss Function`` which is an indicator to learn the input source data and the correct label(value).
   This time, the prediction type is classification.
   In general, the classification can be divided into ``BinaryClassification`` and ``MultiClassification``.
   There are suitable loss functions for each case.

   Oh, we need to consider it at first?

   No, you are using Marquetry so let's leave such a troublesome matter to the framework!

   We use :func:`marquetry.functions.classification_cross_entropy` which detects and chooses the classification type and
   loss function automatically.

   .. code-block:: python

      loss_func = mq.functions.classification_cross_entropy

How to learn the data?
***********************
The model learns the data and the corresponding correct by updating the parameters.
What is the update indicator? That is exactly what it is ``Loss``.

Try to remember, when the model is fitted to the data, the ``Loss`` is reduced.
In other words, the ``Loss`` is reduced, and the model will fit the data.

Now, we prepare the data and model and loss_function so the next component is the last and important thing,
which is called ``Optimizer``.

The model fitting is called ``Optimize the model``, so the optimizer is the update function.
Internally, model fitting reduces the loss by the gradient of the loss for each parameter.
Optimizer updates the parameter following the gradient to reduce the loss.

To resolve some issues, handled and thousands of optimizers have been presented so far.
In this time, we use SGD :class:`marquetry.optimizers.SGD` which is the most simple optimizer.

:The formula is: previous_param -= learning_rate(small constant value) * the corresponding gradient

What is the gradient? Okay, I try to explain it briefly!

I planned to not explain this, hahaha but okay, such curiosity is very important!

Try to remember when you were a high school student.

... No! I didn't ask you about your girlfriend when you were in high school! lol

I'd like you to remember mathematics!
Maybe you learned differential. The differential is the tangent slope of the original function.
The tangent slope is the mentioned ``Gradient``.

From a macro perspective,
deep learning(including loss function) can be viewed as a complex function (ten to million dim function).
The slope indicates the direction of the function maximum(at least increasing)
so that the parameter updates to the opposite direction of the gradient, the function result can be decreased.

Try to remember one more, to fit a model to the data, we need to reduce the loss.

Have you figured it out yet?
The Gradient is computed including the loss function,
so if all parameters of the model update in the opposite direction, the loss will be reduced.

The SGD formula follows this mission. Please see again the formula.
 - The formula updates the param by opposite gradient
   (This function computes subtraction of the gradient from previous parameters.)

The ``learning_rate`` prevents large updates, by this mechanism, we can reduce the risk of oscillating the model.
This time we use 0.1 as ``learning_rate``.

3. Prepare optimizer

   .. code-block:: python

      optim = mq.optimizers.SGD(0.1).prepare(model)

.. tip::
   In Marquetry, the model you want to optimize is registered to the optimizer via optimizer's :meth:`prepare`.
   (This is a Marquetry manner, not common knowledge.)

Model Training
~~~~~~~~~~~~~~~~
Finally, we get all we need in this section!
Let's train the model using the created dataset!

Only a few more steps left to do!

We need to decide ``Batch Size`` and ``Epoch`` which are some of hyperparameters for deep learning.

:Batch Size: This means how many records is used for the training at once.
             For Deep Learning, there are 3 methods for this topic. ``batch``, ``mini-batch`` and ``online`` training.

             ``batch``:
                  this method uses all of the data for 1 time training.

             ``mini-batch``:
                  this method uses some sampled data from the original data for 1 time training
                  and the mini-batch combination is changed in each epoch.

             ``online``
                  this method is only 1 record sampled randomly for 1 time training.
                  Generally, the order is changed in each epoch.

.. tip::
   ``batch`` training provides stable training because this method uses all data at once
   so the training is insensitive to the influence of a small noise in data like outlier or so.

   However, ``batch`` needs very large memory space because this method needs all data to be loaded on the memory at once.
   And, the computational load is also increased.

   ``online`` training provides fast learning and low memory usage,
   and can fit real time model update if you need to update the model to fit the real time data like stock value.

   However, ``online`` is sometimes not stable because this method uses only 1 data at 1 training so it sensitive to
   the influence of a small noise, and honestly speaking,
   ``online`` training is slowly comparing 1 data unit compute time with ``batch``.

   ``mini-batch`` is the mixed method of ``batch`` and ``online``.
   This method uses a mini-batch unit at 1 training, each 1 time,
   using a randomly sampled dataset of user defined size(batch size).
   And the size is smaller than the original data size.

   From these specifications, this method insensitive to the influence of noises than ``online`` training and
   smaller than the data size than the ``batch`` training so this can reduce the memory usage.

Currently, almost every case uses ``mini-batch`` training so this time, we use ``mini-batch``.

.. _epoch:

``Epoch`` defines how many times train the data.
In other words, 1 epoch means that all data uses up even the training method is any.

(``batch`` training uses all data at once so this method, ``Epoch`` match with the training times.)

In this time, we use the ``mini-batch`` method with a batch size 32, and the epoch is 2000.

Also, to confirm the progress record the loss, and output the figure per setting interval.
This time, the interval is set as 100.

Let's train the model with data!

.. code-block:: python

   import numpy as np

   batch_size = 32
   total_epoch = 2001
   interval = 100

   iterations = len(target) // batch_size

   sin_data = np.sin(x * np.pi)
   for epoch in range(total_epoch):
       shuffled_index = np.random.permutation(len(target))

       total_loss = 0
       x_0, y_0 = [], []
       x_1, y_1 = [], []

       for iter in range(iterations):
           batch_index = shuffled_index[iter * batch_size:(iter + 1) * batch_size]
           batch_x, batch_t = data[batch_index], target[batch_index]

           y = model(batch_x)
           loss = loss_func(y, batch_t)

           model.clear_grads()
           loss.backward()
           optim.update()

           if epoch % interval == 0:
               y = mq.functions.sigmoid(y)

               total_loss += float(loss.data)

               for index, pred in enumerate(y.data):
                   pred = pred.reshape(-1)

                   if  float(pred[0]) < 0.7:
                       x_0.append(batch_x[index, 0])
                       y_0.append(batch_x[index, 1])

                   else:
                       x_1.append(batch_x[index, 0])
                       y_1.append(batch_x[index, 1])

       if epoch % interval == 0:
           plt.plot(x, sin_data, linestyle="dashed")
           plt.scatter(x_0, y_0, marker="^")
           plt.scatter(x_1, y_1, marker="o", c="m")

           plt.title("epoch: {} / {}, loss: {:.4f}"
                     .format(epoch, total_epoch, total_loss / iterations))
           plt.show()

           print("Epoch: {} / {}, Loss: {:.4f}".format(epoch, total_epoch, total_loss / iterations))


.. centered:: *The output transition*

.. grid:: 3
   :gutter: 2

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_0.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_100.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_200.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_300.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_400.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_600.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_800.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_1000.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_1200.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_1400.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_1600.png

   .. grid-item::

      .. image:: ../../_static/img/nn_sin_2000.png


Your model draws a beautiful sin curve!
Just now, you stepped in to the deep learning world! Congratulation!!!

Welcome to the deep learning world!!

Lastly...
~~~~~~~~~~
This model has only 6 neurons so the expressiveness is limited like just drawing such simple area classification.

Of course, the real problem may be more complex, some problems can't deal with this small model.

However, if you understand these steps, you can expand the model!
Please play with this framework and I hope your journey is all the best!

...how's it going? Impressive, isn't it? lol

Keep in mind, you are only entering the world start line!
We prepare the more practical problem! Let's keep learning!

The next is a prediction of the Titanic Disaster. This problem needs only a fully connected neural network.

``fully connected neural network`` is the same as ``MLP``. So all weapons to resolve the problem are in your hands now!

.. centered:: **Let's go to the practical problem!**

.. button-link:: ../trial_examples/titanic_disaster.html
 :color: info
 :outline:
 :expand:

 Titanic Disaster
