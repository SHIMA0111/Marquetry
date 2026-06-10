"""Train an MLP on FashionMNIST (a real dataset, downloaded on first run).

The dataset (~30MB) is downloaded once and cached under ``~/.marquetry``.

Run:
    python samples/03_fashion_mnist_mlp.py
"""
import numpy as np

import marquetry
import marquetry.functions as funcs
from marquetry import optimizers, transformers
from marquetry.dataloaders import DataLoader
from marquetry.datasets import FashionMNIST
from marquetry.models import MLP

np.random.seed(42)

transform = transformers.Compose([
    transformers.Flatten(),
    transformers.ToFloat(),
    transformers.Normalize(mean=0.0, std=255.0),
])
train_set = FashionMNIST(train=True, transform=transform)
test_set = FashionMNIST(train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=128)

model = MLP([256, 128, 10], activation=funcs.relu, is_dropout=True)
optimizer = optimizers.Adam().prepare(model)

for epoch in range(3):
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

    print("epoch {}: loss={:.4f} accuracy={:.4f}".format(
        epoch + 1, sum_loss / iterations, sum_accuracy / iterations))

# ------------------------------------------------------------------ evaluate
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
correct, total = 0, 0

with marquetry.test_mode(), marquetry.no_backprop_mode():
    for x, t in test_loader:
        prediction = model(x).data.argmax(axis=1)
        correct += int((prediction == t).sum())
        total += len(t)

print("\ntest accuracy: {:.4f}".format(correct / total))
