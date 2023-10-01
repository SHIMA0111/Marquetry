Titanic Disaster Prediction
=============================
Welcome to the Titanic Disaster Prediction!

In this page, we predict Titanic Disaster Survivor using a various passengers' attribute.

Do you know the Titanic?

On April 15, 1912, RMS Titanic, witch was widely considered "unsinkable" at that time,
sank after colliding with an iceberg.
Unfortunately, there weren't enough lifeboats for all passengers. As a result, 1502 of all(2224) passengers and crews
are passed away.
In a later survey, there are some trend between survivors and others.

This prediction is a challenge that build a fit model that "what people were more likely to survive"
using passengers' data(name, age, gender, etc).

(This data obtained from `the Vanderbilt University Department of Biostatistics <http://hbiostat.org/data>`_.)

.. centered:: Let's start to challenge this problem!

Prepare data
~~~~~~~~~~~~~
1. Load data

   We prepared Titanic data as Marquetry built-in dataset. So you can load the dataset easily.

   .. code-block:: python

      import marquetry as mq


      dataset = mq.datasets.Titanic(train=True, train_rate=1., pre_process=False)
      print(dataset.source_shape)
      >>> (1309, 10)

   In Titanic dataset, you can specify the ``train_rate`` by yourself. When ``train_rate`` is 1.0, you get all data.
   So, this dataset has 1309 records.

   And, 10 columns are had as source data (and 1 column is for target data).

   .. tip::
      As a matter of fact, the original data has more 3 data but 2 are leak data and 1 is not exist in the kaggle so
      to be simple, it was deleted by Marquetry.

   Let's check the data.

   .. code-block:: python

      sampled_source, _ = dataset[0]

      print(dataset.source_columns)
      >>> ['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked']

      print(sampled_source)
      >>> [3 'Yasbeck, Mr. Antoni' 'male' 27.0 1 0 '2659' 14.4542 nan 'C']

   Actually, this dataset can't be learned as-is.
   Please focus ``name`` ``sex``, ``ticket``, ``embarked``, these data is expressed as string.
   But in neural network, data is computed by linear transformation so neural network can only treat numeric data.

   Then, what should we do?

   The answer is very simple, such strings are transformed to numeric representing the original data.
   There are many techniques but in here, I introduce most simple 2 techniques.

   First one is called as ``Label Encoding``.

   As the first step, gathering the column unique values, and then, integers assign to the unique values.
   And transform the original column to the integers assigned to the value.

   For example, we had the below column in data we want to make learn model.

   ======  ========
   Index   Classes
   0       A
   1       C
   2       A
   3       B
   4       B
   5       C
   ======  ========

   The unique values of the ``Classes`` are ``{A, B, C}``. And assign number to the unique set like
   ``{A: 0, B: 1, C: 2}``.

   Then, you transform the original value to the number like the below.

   ======  ========
   Index   Classes
   0       0
   1       2
   2       0
   3       1
   4       1
   5       2
   ======  ========

   The data is changed to numeric data however the data meaning isn't changed because 0 is always indicating ``A``
   in this column. Even others are the same.

   Right, very easy and simple, isn't it?

   However, this encord method has some problems. One of the most biggest problems is "What is the magnitude relation?".

   In Label Encoding, the magnitude relation will show up because it just assign number orderly.
   But if the ``Classes`` signifies the class room name, there is no relation in the magnitude
   ("A" class should be neither superior nor inferior others).

   Therefore, such column is often transformed by the next technique.

   The second technique is called as ``One-Hot Encoding``.

   In this method also gather unique values in the column and assign number as the first step.
   However, the next step is different from the ``Label Encoding`` completely.

   As the next step, we prepare the 0-filled matrix as the size of ``record_num``×``unique_num``.
   After that, the number of the assigned number to the value considering as the column index and the corresponding
   record column changing to 1.

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

   In this page, we use these 2 methods to data preparation.

   .. note::
      At first sight, the ``One-Hot Encoding`` seems to be the best option for the no magnitude relationship data.
      However, ``One-Hot Encoding`` has a large problem.

      It is that the one-hot vector can't be controlled the data column size(dimensions).
      If the unique values num are 100,000,000 values, one-hot encoder creates and adds the 100000000 dims data
      to the data.

      Right, if you don't know(can't expect) the unique values num, one-hot encoding can cause
      feature space explosion. This cause also the curse of dimensionality.

   .. tip::
      Curse of Dimensionality is advocated by ``Richard Ernest Bellman`` who is applied mathematician.
      This signifies the computation cost is exponentially increasing following the Dimension of the mathematical space.


   In the Marquetry, you can do this preprocess easily!

   Before it, let's sort the original features.
   Temporary, we ignore the numerical data.

   ``name`` and ``sex`` and ``ticket`` and ``cabin``, and ``embarked`` has no the magnitude relationship so
   these should be transformed to ``One-Hot``.

   In this data, there is no data having the magnitude relationship in the strings columns.

   Then, let's consider about numerical data.
   Firstly, ``age``, ``fare`` are float number, these isn't needed to be encoded to any
   because these can use data as-is.

   - ``sibsp`` means the number of the siblings/Spouses aboard on the Titanic.
   - ``parch`` means the number of the parents/children aboard on the Titanic.

   Therefore, these can be considered as numerical columns.

   For ``pclass``, this indicates passenger class(1st, 2nd, 3rd) which is a proxy for socio-economic class.

   Therefore, ``pclass`` seems to be categorical columns however, this column has a magnitude relationship.
   So we should transform this column to label data.

   .. tip::
      This time, ``pclass`` treats as categorical column and trans it to label data.
      However, some of you think what is there meaning to trans to label.
      Because the original data is also number so you think it could be useful as-is.

      Your thinking is correct so if you can use the data as numerical column too.
      However, in Marquetry, ``pclass`` is set as categorical column built-in.

      Also ``sibsp`` and ``parch`` unique number is limited,
      so that these also can be considered as also Categorical columns.

      If you have such question, you may stand the start point of the feature engineering.

   I have rambled on for quite some time. But the data explanation is up so let's prepare the dataset.
   One more remind, ``pclass`` should be label data and
   ``age``, ``fare``, ``sibsp``, and ``parch`` should be numerical data.
   And others should be one-hot data.
   The category and numerical classification is built-in so you don't need specify it.

   And category columns are assigned to one-hot as default, so you need only specify ``label_encoding_columns``.
   (Also, ``name`` is unique data so this time drop ``name`` column.)

   .. code-block:: python

      dataset = mq.datasets.Titanic(train_rate=0.8, label_columns=["pclass"], drop_columns=["name"])
      test_dataset = dataset.test_data()

   .. tip::
      In Titanic dataset, we suggest to use :meth:`marquetry.datasets.Titanic.test_data` to get test data.

2. Load dataset to dataloader

   DataLoader helps the mini-batch learning to be easy.
   At this time, the ``batch_size`` is 32.

   .. code-block:: python

      batch_size = 32
      shuffle = True

      dataloader = mq.dataloaders.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
      test_dataloader = mq.dataloaders.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

.. centered:: Then you complete preparation. Congratulation!!

Prepare model
~~~~~~~~~~~~~~

1. Create model

   In this time, we try to use Sequential wrapper constructing Fully-connected Neural Network(MLP).
   And, to regularize the learning, we use :class:`marquetry.layers.BatchNormalization`
   and also using :math:`marquetry.functions.relu` as activation function.

   The first Linear transformation has 16 neurons and the output Linear has 1 neurons.

   .. code-block:: python

      model = mq.models.Sequential(mq.layers.Linear(16), mq.layers.BatchNormalization(), mq.functions.relu, mq.layers.Linear(1))

2. Set the model to Optimizer

   We use :class:`marquetry.optimizers.Adam` as optimizer.

   .. code-block:: python

      optim = mq.optimizers.Adam()
      optim.prepare(model)


.. centered:: Now you have all you needed to learn the Titanic dataset! Let's proceed the learning section!

Model fitting
~~~~~~~~~~~~~~

In this time, the ``max_epoch`` is 100 and ``accuracy`` and ``loss`` are used as accuracy indicator.

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
however this model can predict the unknown data almost 80% accuracy.

.. tip::
   Overfitting means the model conforming to the training data excessively.
   Tell you the truth, such model isn't good because almost such model can't predict unknown data correctly.

   In deep learning, the model expression power is very high so overfitting liable occurring.
   To prevent such situation, we consider reducing model expression power or increasing the train data.

   Before now, many method preventing overfitting are developed.
   The :class:`marquetry.layers.BatchNormalization` is one of the methods, and :meth:`marquetry.functions.dropout`
   is also one of the methods.
   And L1/L2/LN regularization is also famous way of preventing overfitting methods.

This data is simple and few so tend to overfit, to prevent this we may be able to use ``up sampling`` or
reducing epoch or reducing neurons or so.

In this section, we don't view such prevent overfitting method, please research and check it out for yourself!

Thank you for your hard work! Now the FNN(Fully-connected Neural Network) example lecture is completed!

FNN is very useful for wide-variety use case. Let's try some problem using Marquetry!

----

Do you want to check more example? Sure! We prepare more example using Marquetry.

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
