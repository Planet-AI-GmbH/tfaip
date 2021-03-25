Load and Validate (LAV)
=======================

This file describes how to evaluate a trained model performed by Load-And-Validate (LAV).
LAV loads an :ref:`exported model<doc.training:Export of Checkpoints and Saved Models>`, applies it on provided data, and finally compute the :ref:`metrics<doc.model:metric>` and :ref:`losses<doc.model:loss>` defined in the :ref:`model<doc.model:model>` but also the output of an optionally provided :ref:`Evaluator<doc.scenario:evaluator>`.

Calling LAV via the command line
--------------------------------

To load and validate a model use a [saved model](http://gitea.planet-ai.de/pai/tf2_aip/wiki/Model-exporting) (e.g. export or best) and the `tfaip-lav` script, usage:

.. code-block::shell

    tfaip-lav --export_dir PATH_TO_SAVED_MODEL

Parameters
~~~~~~~~~~

LAVParams
"""""""""

The ``PipelineParams`` of the ``LAVParams`` are accessed directly via `--lav`, e.g.

.. code-block::shell

    --lav.batch_size 5
    --lav.num_processes 32

The ``DeviceParams`` can be set by, e.g.:

.. code-block::shell

    --lav.device.gpus 0

DataGeneratorParams
"""""""""""""""""""

``tfaip-lav`` allows to adapt the data generator parameters which is useful to change the evaluation data (by default the validation generator when training is used).
For a :ref:`ListFileScenario<doc.scenario:listfilescenario>`, the evaluation list can be changed by:

.. code-block::shell

    --data.lists OTHER_VAL_LIST


Other Parameters
""""""""""""""""

Specify

* ``--run-eagerly`` to run lav in eager-mode (useful for debugging)
* ``--dump`` to dump the targets and predictions to a pickle file (implement a custom `LAV` and `LAV.extract_dump_data` to modify the dump)


Implement a Custom LAV
----------------------

The default ``LAV`` module can be extended to change the default behaviour.
Do not forget the register it in the :ref:`Scenario<doc.scenario:Scenario>` at ``ScenarioBase.lav_cls()`` or ``ScenarioBase.create_lav()``.
