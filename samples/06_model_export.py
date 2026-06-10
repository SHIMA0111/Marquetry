"""Three ways to persist a trained model.

1. ``save_params`` / ``load_params``  -- weights only (npz), needs the model class
2. marquetry archive (``.mq``)        -- graph + weights, loadable without the class,
                                         restored models stay trainable
3. ONNX                               -- run anywhere (requires the ``onnx`` package)

Run:
    python samples/06_model_export.py
"""
import os

import numpy as np

import marquetry
import marquetry.functions as funcs
from marquetry import optimizers, transformers
from marquetry.dataloaders import DataLoader
from marquetry.datasets import Spiral
from marquetry.model_archive import load_archive
from marquetry.models import MLP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------- train quickly
train_set = Spiral(class_num=3, class_data_size=200, random_state=0,
                   transform=transformers.ToFloat(), target_transform=np.argmax)
train_loader = DataLoader(train_set, batch_size=30)

model = MLP([32, 3], activation=funcs.relu, is_dropout=False)
optimizer = optimizers.Adam().prepare(model)

for _ in range(30):
    for x, t in train_loader:
        loss = funcs.softmax_cross_entropy(model(x), t)
        model.clear_grads()
        loss.backward()
        optimizer.update()

sample_input = np.stack([train_set[i][0] for i in range(5)])
with marquetry.test_mode(), marquetry.no_backprop_mode():
    reference = model(sample_input).data

# ------------------------------------------------- 1. weights only (npz)
npz_path = os.path.join(OUTPUT_DIR, "spiral_mlp.npz")
model.save_params(npz_path)

same_architecture = MLP([32, 3], activation=funcs.relu, is_dropout=False)
same_architecture(sample_input)  # initialize the lazy weights before loading
same_architecture.load_params(npz_path)
print("save_params/load_params: restored ->",
      np.array_equal(same_architecture(sample_input).data, reference))

# ----------------------------------- 2. marquetry archive (graph + weights)
archive_path = os.path.join(OUTPUT_DIR, "spiral_mlp.mq")
model.export_archive(sample_input, archive_path)

restored = load_archive(archive_path)  # no model class needed
with marquetry.test_mode(), marquetry.no_backprop_mode():
    archived = restored(sample_input).data
print("marquetry archive (.mq): restored ->", np.array_equal(archived, reference))

# The restored model is still trainable: parameters, autodiff and the
# train/test mode dispatch all survive the round-trip.
print("restored model parameters:", len(list(restored.params())))

# --------------------------------------------------------------- 3. ONNX
try:
    import onnxruntime
except ImportError:
    print("ONNX export skipped: run `pip install onnx onnxruntime` to try it.")
else:
    onnx_path = os.path.join(OUTPUT_DIR, "spiral_mlp.onnx")
    model.export_onnx(sample_input, onnx_path)

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": sample_input})[0]
    print("ONNX via onnxruntime: max |diff| = {:.3e}".format(
        float(np.abs(onnx_output - reference).max())))
