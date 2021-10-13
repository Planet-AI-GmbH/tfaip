Parameter Classes
=================

Parameters are used to control every module in |tfaip| including hyper-paramters of training, the selection of graph, modifying the data pipeline, etc.
Each module is split into a class providing the actual implementation, and a parameter dataclass that only holds its parameters, e.g. ``Trainer`` and ``TrainerParams``.
Each ``Params``-class of `tfaip` is wrapped by ``dataclass`` and ``pai_dataclass`` which allows to easily add new parameters that support typing and will automatically be accessible from the command line.
Command line support is provided by :`paiargparse <https://github.com/Planet-AI-GmbH/paiargparse>`_ which automatically converts a dataclass hierarchy into command line arguments.

In the following, all primary parameter classes are listed

* :ref:`ScenarioBaseParams<doc.scenario:Scenario>`
  * :ref:`DataBaseParams<doc.data:data>`
  * :ref:`ModelParams<doc.model:model>`
  * :ref:`EvaluatorParams<doc.scenario:Evaluator>`
* :ref:`TrainerParams<doc.training:training>`
    * :ref:`DeviceParams<doc.device_config:Device configuration>`
    * :ref:`OptimizerParams<doc.training:optimizer>`
    * :ref:`LearningRateParams<doc.training:Learning rate>`
    * :ref:`WarmstartParams<doc.training:warm-start>`
* :ref:`LAVParams<doc.evaluation:Load and validate (LAV)>`
* :ref:`PredictorParams<doc.prediction:prediction>`


Parameters
----------

All parameters that shall be available during training or loading of a model (e.g., :ref:`resume training<doc.training:resume training>`, or :ref:`LAV<doc.evaluation:load and validate (LAV)>`) must be added to a params class within the hierarchy (see above).

Custom Parameter Classes
~~~~~~~~~~~~~~~~~~~~~~~~

You are allowed to add additional parameter classes which must be wrapped by ``@pai_dataclass`` and ``@dataclass``.
Make sure to use a descriptive name for the name and the field storing that dataclass, e.g.:

.. code-block::python

    @pai_dataclass
    @dataclass
    class BackendParams:
    # This is the additional parameter set that could define a backend (e.g. a convolutional neural net)
    conv_filters: List[int] = field(default_factory=lambda: [8, 16, 32])

    @pai_dataclass
    @dataclass
    class ModelParams(ModelBaseParams):
    # This is the default parameter set for a model
    # other params ...
    backend: BackendParams = field(default_factory=BackendParams)

Subclassing Parameter Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclassing a dataclass will inherit all fields of the parent.
|tfaip| uses this to extend parameters for almost every module.
Note, that this is the reason why every field must have at least a dummy default value.
Required parameters to be set from the command line must be specified by setting the metadata: ``field(..., metadata=pai_meta(required=True))``.

Hidden Parameters
~~~~~~~~~~~~~~~~~

To hide a parameter from the command line, set its field metadata mode to ignore: ``field(..., metadata=pai_meta(mode="ignore")``.
These parameters are considered to be *state* parameters that are set or changed during training (e.g., ``TrainerParams.export_best_state``)
or are set by the scenario to exchange information (e.g., the ``DataParams`` could define ``num_classes`` which are passed to ``ModelParams.num_classes`` during the initialization).
To share parameters implement the ``__post_init__`` function of a parent dataclass if possible, else override ``ScenarioBase.create_model``, ``ScenarioBase.create_data``, etc. to specify a parameter directly before the actual class is instantiated.

Command Line
------------

The parameter hierarchy is parsed and flattened to allow to set the parameters from the command line. The following types are supported, see `here <https://github.com/Planet-AI-GmbH/paiargparse#supported-types>`_ for a full list and `examples <https://github.com/Planet-AI-GmbH/paiargparse#examples>`_:
* Primitive types: ``str``, ``int``, ``float``, ``bool``
* Enums: ``IntEnum``, ``StrEnum``
* Lists: ``List[str]``, ``List[int]``, ``List[float]``, ``List[Enum]``
* Other dataclasses defined with ``@pai_dataclass`` and ``@dataclass``, also in `Dicts` and `Lists`

Naming convention:
* Dataclasses in ``snake`` mode, the *default* (``dc_meta(mode="snake")``) are added as snake mode, e.g. ``--train.batch_size``
* Dataclasses in ``flat`` mode (``pai_meta(mode="flat")``) are added as root parameter, e.g. ``--model`` or ``--trainer``

`Meta data <https://github.com/Planet-AI-GmbH/paiargparse#meta-data>`_ can be specified to for example add a help string or change the mode.

Example:

.. code-block::python

    from dataclasses import dataclass, field
    from typing import List

    @pai_dataclass
    @dataclass
    class ExampleParams:
        example: TYPE = field(default=DEFAULT_VALUE, metadata=pai_meta(help="HELP STR"))

        # For example
        epochs: int = field(default=1000, metadata=pai_meta(help="Number of epochs to train"))
        # Or with factory
        gpus: List[int] = field(default_factory=list, metadata=pai_meta(help="GPUs to use"))


Static Parameters
-----------------

Static parameters are parameters that must be know to create a certain class of ``ModelBase``, ``DataBase``, ``GraphBase``, ``PredictorBase``, ``LAV``, ``Evaluator``, or ``RootGraph``.
Hereto add the parameter as additional argument to the respective ``__init__`` function and override the respective parameter function in ``ScenarioBase``.

See the `how to <https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/howtos/staticmodelparameters>`_ for a usage for ``ModelBase``, ``DataBase``, and ``GraphBase``.

Use this if you want to:

* pass parameters from ``DataBase`` (instantiated) to the ``ModelBase` or ``Evaluator``, e.g., the size of a loaded codec, or the tokenizer.
  See usage in the `ATR example <https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/atr/scenario.py>`_.
