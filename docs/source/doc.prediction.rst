Prediction
==========

After a :ref:`Scenario<doc.scenario:scenario>` was :ref:`trained<doc.training:training>` and `evaluated (LAV)<doc.evaluation:load and validate (LAV)`, a ``Predictor`` is used to apply the resulting model on new data.
Similar to LAV, a prediction can be performed with the command line (``tfaip-predict``) or programmatically.

Each resulting ``Sample`` of the predictor passes the :ref:`post_proc-Pipeline<doc.data:data>`.

Command-Line
------------

.. code-block:: shell

    tfaip-predict --export_dir PATH_TO_EXPORTED_MODEL --data.DATA_GENERATOR_PARAMS --predict.PREDICTOR_PARAMS --pipeline.PIPELINE_PARAMS


``tfaip-predict`` requires at least the path to the :ref:`exported<doc.training:Export of Checkpoints and Saved Models>` model (``--export_dir``).
The ``--data`` flag is used to modify the input data, the ``--prediction`` flag sets up the predictor.
Another optional flag is ``--pipeline`` which allow to specify the ``DataPipelineParams``, e.g., the number of threads or batch size.

Programmatically
----------------

Create a ``Predictor`` by calling ``Scenario.create_predictor(model: str, params: PredictorParams)``.
(Overriding ``Scenario.predictor_cls()`` can be used to customize a ``Predictor``).
The resulting object can be used to ``predict`` given ``DataGeneratorParams``, raw input data ``predict_raw`` which is basically the output of a ``DataGenerator``, to predict a ``DataPipeline`` (``predict_pipeline``), or to predict a ``tf.data.Dataset`` (``predict_database``).
The output of each function is a generator of ``Samples``.

A predicted ``Sample`` holds ``inputs`` and ``outputs``.
Optionally, if available in the data and if the ``predictor_params.include_targets`` flag is set, also the ``targets``.

For example:

.. code-block::python

    predictor = MyScenario.create_predictor("PATH_TO_SAVED_MODEL", PredictorParams())

    # Predict on raw data
    for sample in predictor.predict_raw([Sample(inputs=np.zeros([28, 28]))]):
        print(sample)

    # Predict a data generator
    data = MyDataGenerator()
    for sample in predictor.predict(data):
        print(sample)
