"""Train an MLP on the synthetic Spiral dataset (no download needed).

Shows the standard training loop: Dataset -> DataLoader -> Model -> loss
-> backward -> Optimizer.update, then evaluation in test mode.

Run:
    python samples/02_spiral_classification.py
"""
import numpy as np

import marquetry
import marquetry.functions as funcs
from marquetry import optimizers, transformers
from marquetry.dataloaders import DataLoader
from marquetry.datasets import Spiral
from marquetry.models import MLP

# Spiral targets are one-hot; softmax_cross_entropy expects integer labels.
train_set = Spiral(class_num=3, class_data_size=200, random_state=0,
                   transform=transformers.ToFloat(), target_transform=np.argmax)
test_set = Spiral(class_num=3, class_data_size=100, random_state=1,
                  transform=transformers.ToFloat(), target_transform=np.argmax)

train_loader = DataLoader(train_set, batch_size=30)

model = MLP([32, 32, 3], activation=funcs.relu, is_dropout=False)
optimizer = optimizers.Adam().prepare(model)

for epoch in range(100):
    sum_loss, sum_accuracy, iterations = 0.0, 0.0, 0

    for x, t in train_loader:
        y = model(x)
        loss = funcs.softmax_cross_entropy(y, t)

        model.clear_grads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data)
        sum_accuracy += float(funcs.evaluation.accuracy(y, t).data)
        iterations += 1

    if (epoch + 1) % 20 == 0:
        print("epoch {:3d}: loss={:.4f} accuracy={:.4f}".format(
            epoch + 1, sum_loss / iterations, sum_accuracy / iterations))

# ------------------------------------------------------------------ evaluate
x_test = np.stack([test_set[i][0] for i in range(len(test_set))])
t_test = np.asarray([test_set[i][1] for i in range(len(test_set))])

with marquetry.test_mode(), marquetry.no_backprop_mode():
    y_test = model(x_test)

test_accuracy = float(funcs.evaluation.accuracy(y_test, t_test).data)
print("\ntest accuracy: {:.4f}".format(test_accuracy))
