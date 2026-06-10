"""Classic (non-neural) machine learning: RandomForest and SVM.

Marquetry also ships scratch implementations of classic models sharing the
``fit`` / ``predict`` interface.

Run:
    python samples/05_classic_ml.py
"""
import numpy as np

import marquetry.ml as ml
from marquetry.datasets import Spiral

# ----------------------------------------------------- random forest (3-class)
train_set = Spiral(class_num=3, class_data_size=200, random_state=0)
test_set = Spiral(class_num=3, class_data_size=100, random_state=1)

forest = ml.RandomForest(n_trees=20, target_type="classification", seed=3)
forest.fit(train_set.source, train_set.target)  # one-hot targets are handled

prediction = forest.predict(test_set.source).data
labels = test_set.target.argmax(axis=1)
print("RandomForest accuracy: {:.4f}".format(float((prediction == labels).mean())))

# ----------------------------------------------------------------- svm (binary)
rng = np.random.RandomState(7)
class_a = rng.normal(loc=(-1.5, -1.5), scale=0.8, size=(100, 2))
class_b = rng.normal(loc=(1.5, 1.5), scale=0.8, size=(100, 2))
x = np.vstack([class_a, class_b])
t = np.array([0] * 100 + [1] * 100)

svm = ml.SVM(c=1.0, learn_rate=0.001, epoch=2000)
svm.fit(x, t)

svm_prediction = svm.predict(x).data
print("SVM training accuracy: {:.4f}".format(float((svm_prediction == t).mean())))
