tfaip
=====

This documentation contains all necessary information to effectively work with |tfaip|.
It depicts the structural concepts of |tfaip| and provides a step by step tour throughout all (necessary) components.

|tfaip| offers two **tutorials**:
If being new to |tfaip|, have a look at the `minimal tutorial <https://github.com/Planet-AI-GmbH/tfaip/tree/master/tfaip/scenario/tutorial/min>`_ to get an impression how |tfaip| is structured.
To quickly set up a new scenario copy either a `tutorial <https://github.com/Planet-AI-GmbH/tfaip/tree/master/tfaip/scenario/tutorial>`_ or a `Template <https://github.com/Planet-AI-GmbH/tfaip/tree/master/tfaip/scenario/template>`_.
Templates show the most common functions that must be implemented (search for ``raise NotImplemented`` and comments written in squared brackets) to set up a custom scenario.

.. toctree::
   :maxdepth: 1
   :caption: Documentation

    Installation <doc.installation>
    Parameters <doc.parameters>
    Scenario <doc.scenario>
    Data <doc.data>
    Model <doc.model>
    Graph <doc.graph>
    Training <doc.training>
    Evaluation <doc.evaluation>
    Prediction <doc.prediction>
    Device Configuration <doc.device_config>
    Debugging <doc.debugging>
    Development <doc.development>

.. toctree::
   :maxdepth: 2
   :caption: Tutorials, Template, Examples

    Minimal Tutorial <https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/tutorial/min>
    Full Tutorial <https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/tutorial/full>
    General Template <https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/template/general>
    ListFile Template <https://github.com/Planet-AI-GmbH/tfaip/tree/master/examples/template/listfile>
    Further Examples <https://github.com/Planet-AI-GmbH/tfaip_example_scenarios>

.. toctree::
   :maxdepth: 2
   :caption: Modules

    tfaip.data.* <tfaip.data>
    tfaip.evaluator.* <tfaip.evaluator>
    tfaip.lav.* <tfaip.lav>
    tfaip.model.* <tfaip.model>
    tfaip.scenario.* <tfaip.scenario>
    tfaip.predict.* <tfaip.predict>
    tfaip.trainer.* <tfaip.trainer>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
