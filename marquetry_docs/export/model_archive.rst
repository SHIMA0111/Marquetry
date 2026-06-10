==================
Marquetry Archive
==================

The marquetry archive (recommended extension: ``.mq``) stores the traced
computation graph and the weights in one zip file, without any extra
dependency. Unlike :meth:`marquetry.Layer.save_params` (weights only),
the archive can be loaded back **without the original model class**,
and the restored model supports both inference and further training.

.. code-block:: python

   import numpy as np
   import marquetry
   from marquetry.model_archive import load_archive

   model = marquetry.models.MLP([32, 10])
   sample = np.random.randn(4, 16).astype(np.float32)

   model.export_archive(sample, "model.mq")

   restored = load_archive("model.mq")   # the MLP class is not needed
   y = restored(sample)                  # inference, backward and updates all work

.. autofunction:: marquetry.model_archive.save_archive

.. autofunction:: marquetry.model_archive.load_archive

.. autoclass:: marquetry.model_archive.GraphModel
   :members:

.. autoclass:: marquetry.model_archive.ArchiveError
