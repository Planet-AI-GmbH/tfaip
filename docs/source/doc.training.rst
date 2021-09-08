Training
========

This section deals about how to :ref:`start <doc.training:launch training>` or :ref:`resume<doc.training:resume training>` via the command line and how to modify the :ref:`training hyper-parameters<doc.training:parameters>`.

Command-Line
------------

Use the command line to launch or resume a training.

Launch Training
~~~~~~~~~~~~~~~

Training of a scenario is performed by the ``train.py`` script which is launched by ``tfaip-train``:

.. code-block:: shell

    tfaip-train {SCENARIO_MODULE} PARAMS

The ``SCENARIO_MODULE`` must be located in the ``PYTHONPATH``.
Either give the full name to scenario class, the scenario file, or the parent module:

.. code-block:: shell

    tfaip-train tfaip.scenario.tutorial.full.scenario:TutorialScenario ...
    tfaip-train tfaip.scenario.tutorial.full.scenario ...
    tfaip-train tfaip.scenario.tutorial.full ...

The first option is required if there are multiple Scenarios located in a single python file.

Resume Training
~~~~~~~~~~~~~~~

Training can be resumed using ``python resume_training.py`` or ``tfaip-resume-training``.
Specify a checkpoint and that's it, e.g.:

.. code-block:: shell

    tfaip-train tfaip.scenario.tutorial.full --trainer.output_dir model_output
    # CANCELLED
    tfaip-resume-training model_output

Resume training and adapt parameters (e.g., extend the ``epochs``): This is not supported directly, yet, however you can manipulate the ``trainer_params`` in the ``output_dir`` and manually adapt the settings.

Debugging Training
~~~~~~~~~~~~~~~~~~

To use the debugger call ``tfaip/scripts/train.py`` instead of ``tfaip-train`` in a run-configuration (e.g., in PyCharm).
See :ref:`here<doc.debugging:Debugging>` for further information.

Validation During Training
~~~~~~~~~~~~~~~~~~~~~~~~~~

Validation during training is used to compute and print the validation performance and to compute the best performing model (see :ref:`early-stopping<doc.training:early-stopping>`).
Validation is optional and will be performed on the provided validation list, e.g. ``--val.lists`` for a :ref:`ListFileScenario<doc.scenario:ListFileScenario>`.

Default
"""""""

By default, validation will be performed by keras and thereto computes the provided loss and metrics of the respective :ref:`Model<doc.model:Model>`.
The default validation is run every epoch which is defined by the ``TrainerParams.val_every_n`` parameter which defaults to 1.

Note that the default validation will be performed on the training graph.
Therefore, if the scenario requires a different prediction graph (e.g. for a Encoder-Decoder-Model) use the extended validation.

Extended
""""""""

It is possible to run a clean validation via [LAV](10_evaluation.md) during training, i.e., loading the current prediction graph and applying it on the validation list.
To enable set the ``TrainerParams.lav_every_n`` parameter to specify on which epochs to run (e.g., ``--trainer.lav_every_n=1``. First and last epochs are always validated if LAV is enabled).
By default, LAV will use the validation generator, but this can be overwritten in the respective ``TrainingPipelineGeneratorParams``.
Note, a :ref:`ListFileScenario<doc.scenario:listfilescenario>` already provides an additional parameter ``lav.lists`` which defaults to ``val.lists``.

LAV will then evaluate all metrics (including the ones of the :ref:`Evaluator<doc.scenario:evaluator>`) and print them (also to the :ref:`Tensorboard<doc.training:tensorboard>`).

Parameters
----------

The most important parameter during training is the ``output_dir`` which defines where to store the log, the exported models, and checkpoints.
Set with ``--trainer.output_dir MY_OUTPUT_DIR``.
Other parameters are introduced in the following.

Logging
~~~~~~~

``tfaip`` uses the ``logging`` module of python.

Set up the log level using the ``TFAIP_LOG_LEVEL`` environment variable, e.g. ``TFAIP_LOG_LEVEL=debug``.
By default, the log level is set to ``info``.

Logging events written to ``logging`` are printed and also written to the ``train.log``.

Learning rate
~~~~~~~~~~~~~

The learning rate can be adapted using the ``trainer.learning_rate`` field which must be set to a ``LearningRateParams`` structure.
The `LearningRateParams` always provide a ``lr``-field to modify the overall learning rate which defaults to ``0.001``.

Example: ``--learning_rate.lr 0.001``.

To change the schedule, set the learning rate field directly: ``--learning_rate ExponentialDecay``.
The parameters of the schedule can be set similarly to above, e.g., ``--learning_rate.decay 0.95``.

Optimizer
~~~~~~~~~

The optimizer of the trainer can be changed and adapted via the ``trainer.optimizer`` field which is a ``OptimizerParams`` structure.
|tfaip| supports different optimizers by default: ``Adam``, ``Adamax``, ``AdaBelief``, ``RMSprop``, ``SGD``. Each one comes with is custom parameters.

Example: ``--optimizer Adamax``.

To adap the parameters of the optimizer call, e.g., ``--optimizer.epsilon 1e-7``

Gradient-Clipping
"""""""""""""""""

Each optimizer supports gradient-clipping based on the `Tensorflow-Optimizer <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer>`_: ``clip_value``, ``clip_norm``, ``clip_global_norm``.

Example: ``--optimizer.clip_global_norm 5``.

Weight-Decay
""""""""""""

A global weight decay (applied to all weights) is provided by the ``Adam`` and ``SGD`` optimizer with their additional ``weight_decay`` field which defaults to ``0.0``.
Alternatively, you can implement weight-decay directly when define layers, as recommended by `Tensorflow <https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#combined_l2_dropout>`_.

Example: ``--optimizer.weight_decay 0.00002``.

EMA-Weights
~~~~~~~~~~~

|tfaip| supports to compute an exponential moving average (EMA) on the weights which is enabled by ``trainer.calc_ema``.
In many cases this leads to improved results on the drawback that more GPU-memory is required during training.
Note that the exported models are always saved along with the EMAs, while checkpoints comprise both the EMAs and the last weights, i.e. the current state of the trainer.
Furthermore, EMA weights are used for validation.

The parameter expects the ema rate, if ``-1``, the rate is computed automatically.

Example: ``--trainer.calc_ema 0.99``

No train scope
~~~~~~~~~~~~~~

Use ``trainer.no_train_scope`` to pass a regex which defines which layers to exclude from training.
Note, if a parent layer is matched, all children will also be not trained.

Example: ``--trainer.no_train_scope '.*conv.*'``.

Warm-Start
~~~~~~~~~~

Warm-starting a model before training with predefined weights is supported.
See ``WarmstartParams`` for all options.

Example: ``--warmstart.model PATH_TO_WARMSTART_MODEL``

Devices
~~~~~~~

See :ref:`DeviceConfig <doc.device_config:Device Configuration>` which is set at ``trainer.device``

Example: ``--device.gpus 0 1``.

Early-Stopping
~~~~~~~~~~~~~~

Setting up early stopping via the ``EarlyStoppingParams`` in ``trainer.early_stopping`` allows the trainer to automatically determine on different constraints when to stop.
Monitoring is based on the best model determined by the :ref:`settings of the model<doc.model:logging the best model>`.

To enable early stopping set ``n_to_go`` to the number of epochs after which to stop training if no improvements was achieved.
To modify not to test every epoch, increase the ``frequency`` (defaults to ``1``, i.e., test every epoch).

Furthermore, you can specify a ``lower_threshold`` and an ``upper_threshold``.
Depending on the ``mode`` during setup, the monitored value must be at least one threshold for early stopping to start, or if the other one is reached, training is stopped immediately.
For example if monitoring the ``max`` ``accurary`` and ``upper_threshold=0.99`` and ``lower_threshold=0.5``, training will not stop until an ``accuracy`` of at least ``0.5`` is reached, then early stopping could kick in.
However, if ``accuracy`` exceeds or equals to ``0.99`` training is stopped immediately. Therefore, setting ``upper_threshold=1`` is sensible if monitoring accuracies.


Export of Checkpoints and Saved Models
--------------------------------------

During training several models/weights are logged:

* ``checkpoint``: stores the complete state (weights and optimizer) used to resume the training.
* ``best``: saved model, that stores the best model on the validation set
* ``export``: saved model, the final state of the model (last model)

There are two formats:

* checkpoint: only the weights are stored. Can only be used to continue the training or for :ref:`warm-start<doc.training:warm-start>`
* saved model (serving): the model and weights are stored. Can be used in LAV and from other apis (e.g., java). Can also be used for :ref:`warm-start<doc.training:warm-start>`

Customizing the Exported Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the exported graph uses all available inputs and outputs tensors, i.e. the inputs and outputs of the :ref:`created graph<doc.graph:graph>`.
To modify this, or to add additional graphs for export, override the `_export_graphs<doc.model:exporting additional graphs>` function of the `ModelBase`.

Tensorboard
-----------

Training can be monitored by the `TensorBoard <https://www.tensorflow.org/tensorboard/>`_.
Hereto, |tfaip| automatically stores the metrics and losses on the train, validation, and lav datasets to the ``output_dir`` which can be displayed by the TensorBoard.
Launch the tensorboard via ``tensorboard --log_dir PATH_TO_THE_MODELS --bind_all``.

Additional output such as images or PR-curves can be setup in the :ref:`model <doc.model:tensorboard>`.

Benchmarking
------------


Profiling Training
~~~~~~~~~~~~~~~~~~

The full training can be profiled using the `Tensorboard`:

* Install requirement ``tensorboard_plugin_profile``
* Set ``--trainer.profile True``


Benchmarking the Input Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quite often, the bottleneck is not the model but the input data pipeline that is not able to produce enough samples per second.

Replace ``tfaip-train`` with ``tfaip-benchmark-input-pipeline`` and run to profile the number of samples per second.
By default the benchmark will run infinitely, so terminate to process to stop the benchmark.
