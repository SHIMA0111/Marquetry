"""Autograd basics: build a computation graph by just writing math, then backprop.

Marquetry records the computation graph while the forward computation runs
(define-by-run), so any Python control flow works and gradients come for free.

Run:
    python samples/01_autograd_basics.py
"""
import numpy as np

import marquetry
import marquetry.functions as funcs


# --------------------------------------------------------------- scalar case
# y = x^3 + e^x  ->  dy/dx = 3x^2 + e^x
x = marquetry.array(2.0, name="x")
y = x ** 3 + funcs.exp(x)
y.backward()

analytic = 3 * 2.0 ** 2 + np.exp(2.0)
print("y           :", float(y.data))
print("dy/dx       :", float(x.grad.data))
print("analytic    :", analytic)

# --------------------------------------------------------------- tensor case
# A tiny linear regression step written as plain math.
np.random.seed(0)
inputs = marquetry.array(np.random.randn(8, 3).astype(np.float32), name="inputs")
weight = marquetry.array(np.random.randn(3, 1).astype(np.float32), name="weight")
targets = marquetry.array(np.random.randn(8, 1).astype(np.float32), name="targets")

prediction = funcs.matmul(inputs, weight)
loss = funcs.mean_squared_error(prediction, targets)
loss.backward()

print("\nloss        :", float(loss.data))
print("weight.grad :", weight.grad.data.ravel())

# Gradients accumulate, so clear them before the next backward pass.
weight.clear_grad()

# ------------------------------------------------------ disabling the graph
# Inside no_backprop_mode() nothing is recorded: faster and lighter for
# inference, but backward() is impossible there.
with marquetry.no_backprop_mode():
    silent = inputs * 2.0
print("\ncreator in no_backprop_mode:", silent.creator)  # -> None
