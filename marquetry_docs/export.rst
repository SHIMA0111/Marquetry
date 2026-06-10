=============
Model Export
=============

Marquetry can persist a trained model in three ways:

 - :func:`marquetry.onnx_export.export_onnx` --
   export the traced inference graph as an **ONNX** model which runs anywhere
   ONNX Runtime works.
 - :func:`marquetry.model_archive.save_archive` --
   save the graph and the weights together as a **marquetry archive** (``.mq``).
   The archive can be loaded back with :func:`marquetry.model_archive.load_archive`
   without the original model class, and the restored model stays trainable.
 - :meth:`marquetry.Layer.save_params` -- plain npz weight checkpoints
   (weights only; you need the original model class to restore them).

Both exporters trace the model by running one forward pass with sample inputs
in test mode, so the exported graph reflects the inference behavior
(e.g. dropout is excluded).

.. toctree::
   :maxdepth: 1
   :hidden:

   export/onnx_export
   export/model_archive

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: ONNX Export
      :link: export/onnx_export
      :link-type: doc
      :text-align: center

   .. grid-item-card:: Marquetry Archive
      :link: export/model_archive
      :link-type: doc
      :text-align: center
