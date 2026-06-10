============
ONNX Export
============

Export a model as an `ONNX <https://onnx.ai/>`_ inference graph.
This feature requires the ``onnx`` package (``pip install "marquetry[onnx]"``),
and the exported model can be verified or executed with
`ONNX Runtime <https://onnxruntime.ai/>`_.

.. code-block:: python

   import numpy as np
   import marquetry

   model = marquetry.models.MLP([32, 10])
   sample = np.random.randn(4, 16).astype(np.float32)

   model.export_onnx(sample, "model.onnx")

.. autofunction:: marquetry.onnx_export.export_onnx

.. autoclass:: marquetry.onnx_export.ONNXExportError
