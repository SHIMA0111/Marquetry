Titanic Disaster Prediction
=============================
Welcome to the Titanic Disaster Prediction!

On this page, we predict Titanic Disaster Survivor using various passengers' attributes.

Do you know the Titanic?

On April 15, 1912, RMS Titanic, which was widely considered "unsinkable" at that time,
sank after colliding with an iceberg.
Unfortunately, there weren't enough lifeboats for all passengers. As a result, 1502 of all(2224) passengers and crews
are passed away.
In a later survey, there are some trends between survivors and others.

This prediction is a challenge that builds a fit model that "what people were more likely to survive"
using passengers' data(name, age, gender, etc).

(This data was obtained from `the Vanderbilt University Department of Biostatistics <http://hbiostat.org/data>`_.)

.. centered:: Let's start to challenge this problem!

Prepare data
~~~~~~~~~~~~~
1. Load data

   We prepared Titanic data as the Marquetry built-in dataset. So you can load the dataset easily.

   .. code-block:: python

      import marquetry as mq


      dataset = mq.datasets.Titanic(train=True, train_rate=1., pre_process=False)
      print(dataset.source_shape)
      >>> (1309, 10)

   In the Titanic dataset, you can specify the ``train_rate`` by yourself. When ``train_rate`` is 1.0, you get all data.
   So, this dataset has 1309 records.

   And, 10 columns are had as source data (and 1 column is for target data).

   .. tip::
      As a matter of fact, the original data has more 3 data but 2 are leak data and 1 isn't in the Kaggle so
      to be simple, it was deleted by Marquetry.

   Let's check the data.

   .. code-block:: python

      sampled_source, _ = dataset[0]

      print(dataset.source_columns)
      >>> ['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked']

      print(sampled_source)
      >>> [3 'Yasbeck, Mr. Antoni' 'male' 27.0 1 0 '2659' 14.4542 nan 'C']

   Actually, this dataset can't be learned as-is.
   Please focus on ``name`` ``sex``, ``ticket``, ``embarked``, these data is expressed as a string.
   But in neural network, data is computed by linear transformation so neural network can only treat numeric data.

   Then, what should we do?

   The answer is very simple, such strings are transformed into numeric representing the original data.
   There are many techniques but here, I introduce the most simple 2 techniques.

   The first one is called as ``Label Encoding``.

   As the first step, gathering the column's unique values, and then, integers are assigned to the unique values.
   And transform the original column to the integers assigned to the value.

   For example, we had the below column in data we want to make a learn model.

   ======  ========
   Index   Classes
   0       A
   1       C
   2       A
   3       B
   4       B
   5       C
   ======  ========

   The unique values of the ``Classes`` are ``{A, B, C}``. And assigned number to the unique set is
   ``{A: 0, B: 1, C: 2}``.

   Then, you transform the original value to the number like below.

   ======  ========
   Index   Classes
   0       0
   1       2
   2       0
   3       1
   4       1
   5       2
   ======  ========

   The data is changed to numeric data however the data meaning isn't changed because 0 always indicates ``A``
   in this column. Even others are the same.

   Right, very easy and simple, isn't it?

   However, this encode method has some problems. One of the biggest problems is "What is the magnitude relation?".

   In Label Encoding, the magnitude relation will show up because it just assigns number orderly.
   But if the ``Classes`` signifies the classroom name, there is no relation in the magnitude
   ("A" class should be neither superior nor inferior to others).

   Therefore, such a column is often transformed by the next technique.

   The second technique is called as ``One-Hot Encoding``.

   This method also gathers unique values in the column and assigns number as the first step.
   However, the next step is different from the ``Label Encoding`` completely.

   As the next step, we prepare the 0-filled matrix as the size of ``record_num``×``unique_num``.
   After that, the labeled number is considered as the column index,
   and the corresponding cell in the matrix changes to 1.

   Using the same classes column, the one-hot is the below

   ======  ==================  ==================  ==================
   Index   Classes_A(index:0)  Classes_B(index:1)  Classes_C(index:2)
   0       1                   0                   0
   1       0                   0                   1
   2       1                   0                   0
   3       0                   1                   0
   4       0                   1                   0
   5       0                   0                   1
   ======  ==================  ==================  ==================

   .. tip::
      The ``One-Hot`` means one of the record data is hot(1), and others are 0.
      Therefore, one-hot encoding creates the vector for each record following the ``One-Hot`` definition.

   This method provides ``One-Hot`` vector so now we can no longer be misled by the magnitude relationship.

   In this page, we use these 2 methods for data preparation.

   .. note::
      At first sight, the ``One-Hot Encoding`` seems to be the best option for the no magnitude relationship data.
      However, ``One-Hot Encoding`` has a large problem.

      It is that the one-hot vector's size(dimensions) can't be controlled.
      If the unique values num are 100,000,000 values, one-hot encoder creates and adds the 100000000 dims data
      to the data.

      Right, if you don't know(can't expect) the unique values num, one-hot encoding can cause
      feature space explosion. This cause also the curse of dimensionality.

   .. tip::
      The ``Curse of Dimensionality`` is advocated by ``Richard Ernest Bellman`` who is an applied mathematician.
      This signifies the computation cost is exponentially increasing following the Dimension of the mathematical space.


   In the Marquetry, you can do this preprocess easily!

   Before that, let's sort out the original features.
   Temporarily, we ignore the numerical data.

   ``name`` and ``sex`` and ``ticket`` and ``cabin``, and ``embarked`` has no the magnitude relationship so
   these should be transformed into ``One-Hot``.

   In this data, there is no data having the magnitude relationship in the strings columns.

   Then, let's consider numerical data.
   Firstly, ``age``, ``fare`` are float numbers, these don't need to be encoded to any
   because these can use data as-is.

   - ``sibsp`` means the number of siblings/Spouses aboard the Titanic.
   - ``parch`` means the number of parents/children aboard the Titanic.

   Therefore, these can be considered as numerical columns.

   For ``pclass``, this indicates passenger class(1st, 2nd, 3rd) which is a proxy for socio-economic class.

   Therefore, ``pclass`` seems to be a categorical column however, this column has a magnitude relationship.
   So we should transform this column to label data.

   .. tip::
      This time, ``pclass`` treats as a categorical column and trans to label data.
      However, some of you think about what is there meaning to trans to label.
      Because the original data is also a number so you think it could be useful as-is.

      Your thinking is correct so you can use the data as numerical column too.
      However, in Marquetry, ``pclass`` is set as categorical column built-in.

      Also ``sibsp`` and ``parch`` unique number is limited,
      so that these also can be considered as a Categorical columns.

      If you have such a question, you may stand the starting point of feature engineering.

   I have rambled on for quite some time. But the data explanation is up so let's prepare the dataset.
   One more reminder, ``pclass`` should be labeled data and
   ``age``, ``fare``, ``sibsp``, and ``parch`` should be numerical data.
   And others should be one-hot data.
   The category or numerical detection is built-in so you don't need to specify it.

   Category columns are assigned to one-hot as default, so you need to only specify ``label_encoding_columns``.
   (Also, ``name`` is unique data so this time drops ``name`` column.)

   .. code-block:: python

      dataset = mq.datasets.Titanic(train_rate=0.8, label_columns=["pclass"], drop_columns=["name"])
      test_dataset = dataset.test_data()

   .. tip::
      In Titanic dataset, we suggest to use :meth:`marquetry.datasets.Titanic.test_data` to get test data.

2. Load dataset to dataloader

   DataLoader helps the mini-batch learning to be easy.
   In this time, the ``batch_size`` is 32.

   .. code-block:: python

      batch_size = 32
      shuffle = True

      dataloader = mq.dataloaders.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
      test_dataloader = mq.dataloaders.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

.. centered:: Then you complete preparation. Congratulation!!

Prepare model
~~~~~~~~~~~~~~

1. Create a model

   In this time, we try to use a Sequential wrapper to construct Fully-connected Neural Network(MLP).
   And, to regularize the learning, we use :class:`marquetry.layers.BatchNormalization`
   and also using :math:`marquetry.functions.relu` as an activation function.

   The first Linear transformation has 16 neurons and the output Linear has 1 neuron.

   .. code-block:: python

      model = mq.models.Sequential(mq.layers.Linear(16), mq.layers.BatchNormalization(), mq.functions.relu, mq.layers.Linear(1))

2. Set the model to Optimizer

   We use :class:`marquetry.optimizers.Adam` as optimizer.

   .. code-block:: python

      optim = mq.optimizers.Adam()
      optim.prepare(model)


.. centered:: Now you have all you need to learn the Titanic dataset! Let's proceed the learning section!

Model fitting
~~~~~~~~~~~~~~

In this time, the ``max_epoch`` is 100, and ``accuracy`` and ``loss`` are used as accuracy indicators.

.. code-block:: python

   max_epoch = 100
   for epoch in range(max_epoch):

       sum_acc, sum_loss = 0, 0
       iterations = 0
       for data, label in dataloader:
           iterations += 1

           y = model(data)
           loss = mq.functions.classification_cross_entropy(y, label)

           acc = mq.functions.evaluation.binary_accuracy(y, label)

           model.clear_grads()
           loss.backward()
           optim.update()

           sum_loss += float(loss.data)
           sum_acc += float(acc.data)

       print("{} / {} epoch | loss: {:.4f} | accuracy: {:.4f}"
             .format(epoch + 1, max_epoch, sum_loss / iterations, sum_acc / iterations))

   test_acc, test_loss = 0, 0
   iterations = 0

   with mq.test_mode():
       for data, label in test_dataloader:
           iterations += 1

           y = model(data)

           test_loss += float(mq.functions.classification_cross_entropy(y, label).data)
           test_acc += float(mq.functions.evaluation.binary_accuracy(y, label).data)

   print("Test data | loss: {:.4f} | accuracy: {:.4f}".format(test_loss / iterations, test_acc / iterations))

The result is

.. code-block::

   1 / 100 epoch | loss: 0.6081 | accuracy: 0.6747
   2 / 100 epoch | loss: 0.4962 | accuracy: 0.7347
   ...
   100 / 100 epoch | loss: 0.0101 | accuracy: 0.9941

   Test data | loss: 0.7382 | accuracy: 0.7969

From this result, this model overfit the train data,
however, this model can predict the unknown data with almost 80% accuracy.

.. tip::
   Overfitting means the model conforms to the training data excessively.
   To tell you the truth, such models aren't good because almost such models can't predict unknown data correctly.

   In deep learning, the model expression power is very high so overfitting liable.
   To prevent such a situation, we consider reducing model expression power or increasing the train data.

   Before now, many methods preventing overfitting have been developed.
   The :class:`marquetry.layers.BatchNormalization` is one of the methods, and :meth:`marquetry.functions.dropout`
   is also one of the methods.
   L1/L2/LN regularization is also a famous way of preventing overfitting methods.

This data is simple and few tend to overfit, to prevent this we may be able to use ``up sampling`` or
reducing epochs or reducing neurons or so.

In this section, we don't view such prevent overfitting method, please research and check it out for yourself!

Thank you for your hard work! Now the FNN(Fully connected Neural Network) example lecture is completed!

FNN is very useful for a variety of use cases. Let's try some problems using Marquetry!

----

Do you want to check more examples? Sure! We prepare more example using Marquetry.

Do you want to check image classification?:
   .. button-link:: ./mnist_cnn.html
      :color: info
      :outline:
      :expand:

      MNIST classification

Would you like to check time-series data?:
   .. button-link:: ./sequential_data_rnn.html
      :color: info
      :outline:
      :expand:

      Trigonometric toy problem
