Device Configuration
====================

The definition which devices are visible and used by |tfaip| and thus Tensorflow is defined via the ``DeviceConfigParams``.
They are used during :ref:`training<doc.training:devices>`, :ref:`validation<doc.evaluation:LAVParams>`, and prediction.
The structure also allows to modify the ``DistributionStrategy`` when training on several GPUs.

Selecting GPUs
--------------

By default, |tfaip| will not use any GPU.
Modify the used GPUs by setting the ``DeviceConfigParams.gpus`` flag which expects a list of GPU indices, e.g.

.. code-block::

    --trainer.device.gpus 0 2

to use the GPUs with index 0 and 2.

Alternatively, if the environment variable ``CUDA_VISIBLE_DEVICES`` is set, |tfaip| will use these GPUs for training.

Multi-GPU setup
---------------

Set the ``DeviceConfigParams.dist_strategy`` flag to ``mirror`` or ``central_storage`` to define how to use multiple devices.
