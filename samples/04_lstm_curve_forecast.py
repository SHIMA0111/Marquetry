"""Forecast a trigonometric curve with an LSTM (truncated BPTT).

The LSTM keeps its hidden state between calls, so the loss is accumulated for
``bptt_length`` steps before backprop, and the graph is cut afterwards with
``unchain_backward`` (truncated backpropagation through time).

Run:
    python samples/04_lstm_curve_forecast.py
"""
import numpy as np

import marquetry
import marquetry.functions as funcs
import marquetry.layers as layers
from marquetry import Model, optimizers
from marquetry.dataloaders import SeqDataLoader
from marquetry.datasets import TrigonometricCurve

np.random.seed(0)


class Forecaster(Model):

    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = layers.LSTM(hidden_size)
        self.out = layers.Linear(1)

    def reset_state(self):
        self.lstm.reset_state()

    def forward(self, x):
        return self.out(self.lstm(x))


train_set = TrigonometricCurve(train=True)  # noisy sin curve
train_loader = SeqDataLoader(train_set, batch_size=30)

model = Forecaster(hidden_size=32)
optimizer = optimizers.Adam().prepare(model)
bptt_length = 30

for epoch in range(5):
    model.reset_state()
    epoch_loss, updates = 0.0, 0
    loss, count = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = loss + funcs.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == train_loader.max_iters:
            model.clear_grads()
            loss.backward()
            loss.unchain_backward()  # cut the graph: truncated BPTT
            optimizer.update()

            epoch_loss += float(loss.data)
            updates += 1
            loss = 0

    print("epoch {}: loss={:.6f}".format(epoch + 1, epoch_loss / updates))

# ------------------------------------------------------- forecast a cos curve
# The test split is a clean cos curve: predict each next value from the
# current one while the LSTM carries the context.
test_set = TrigonometricCurve(train=False)

model.reset_state()
predictions = []
with marquetry.test_mode(), marquetry.no_backprop_mode():
    for index in range(len(test_set)):
        x, _ = test_set[index]
        y = model(x[np.newaxis, :])
        predictions.append(float(y.data[0, 0]))

targets = test_set.target.ravel()
mse = float(np.mean((np.asarray(predictions) - targets) ** 2))
print("\ntest MSE (cos curve): {:.6f}".format(mse))
