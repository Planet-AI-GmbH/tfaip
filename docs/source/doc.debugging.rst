Debugging
=========

Sooner or later, there will be a point during development where debugging is essential.
Since |tfaip| uses Tensorflow 2 which allows for eager execution, debugging is drastically simplified as each computation of the graph can be manually traced by a debugger.
Unfortunately, there are some rare circumstances that lead to code that runs in eager but not in graph mode.
Usually, the reason is that operations are used that are only allowed in eager-mode.

This file shows how to efficiently debug the :ref:`data-pipeline<doc.debugging:data-pipeline>`, the :ref:`model<doc.debugging:model>`, its :ref:`graph<doc.debugging:graph>`, :ref:`loss<doc.debugging:loss>`, and :ref:`metrics<doc.debugging:metric>`, and how to :ref:`profile<doc.debugging:profiling>` using the Tensorboard which helps to detect bottlenecks.


Data-Pipeline
-------------

Debugging of the data-pipeline is usually the first step since this helps to verify the data integrity.
While ``tf.data.Datasets`` can not be debugged easily, the |tfaip| pipeline based on :ref:`DataProcessors<doc.data:data>` can easily be debugged.

Do the following:

* Set up a data-:ref:`test<doc.installation:tests>` for your scenario and :ref:`Data<doc.data:data>` class
* Disable all multiprocessing (setting ``run_parallel=False`` in the ``pre_proc`` and ``post_proc`` parameters of the ``DataParams``).
* Create a ``DataPipeline``.
* Enter the ``DataPipeline`` to obtain a ``RunningDataPipeline``
* Call ``generate_input_samples`` which will return a Generator of samples which are the un-batched input of the ``tf.data.Dataset``.
* Optionally call ``input_dataset().as_numpy_iterator()`` to access the outputs of the ``tf.data.Dataset``. Note that this makes debugging of the pipeline impossible this ``tf.data.Dataset`` is accessed.
  Use this only if you want to verify the batched and padded outputs of the dataset not to debug the data-pipeline itself.

Here is an example for the Tutorial:

.. code-block:: python

    class TestTutorialData(unittest.TestCase):
        def test_data_loading(self):
            trainer_params = TutorialScenario.default_trainer_params()
            data = TutorialData(trainer_params.scenario.data)
            with trainer_params.gen.train_data(data) as rd:
                for sample in rd.generate_input_samples(auto_repeat=False):
                    print(sample)  # un-batched, but can be debugged

                # or
                for sample in rd.input_dataset(auto_repeat=False).as_numpy_iterator():
                    print(sample)  # batched and prepared (inputs, targets) tuple, that can not be debugged. Use prints.

Note that ``generate_input_samples()`` will run infinitely for the ``train_data`` which is why ``auto_repeat=False`` is set to only generate an epoch of data.

Model
-----

To allow for debugging of the model, enable the eager mode (pass ``--trainer.force_eager True`` during :ref:`training<doc.training:training>`, or ``--lav.run_eagerly True`` during :ref:`LAV<doc.evaluation:load and validate (lav)>`)).
Now, the full computations of the graph can be followed.

Graph
~~~~~

During training, additionally pass ``--scenario.debug_graph_construction``.
This will once evaluate the (prediction) graph and compute the :ref:`loss<doc.debugging:loss>` and :ref:`metrics<doc.debugging:metric>` on real data.
It is recommended to use this flag if any error occurs in the graph during construction.

Loss
~~~~

Losses can be fully debugged in eager mode.

Metric
~~~~~~

Metrics of the model can be fully debugged in eager mode.
Also metrics defined in the :ref:`Evaluator<doc.scenario:evaluator>` can always be debugged since they run in pure Python.

Profiling
---------

Profiling is useful to detect bottlenecks in a scenario that slow down training.
Pass the ``--trainer.profile True`` flag to write the full profile of the training (graph mode required) to the Tensorboard.
Also have a look at the `official documentation <https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras#use_the_tensorflow_profiler_to_profile_model_training_performance>`_.

Optimizing the input pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many cases, the input pipeline to too slow to generate samples for the model. However, there are several parameters for tweaking:

* First, enable parallel processing of the pipeline by setting ``run_parallel`` to ``True``.
* Increase the number of threads for the pipeline ``--train.num_processes 16``.
* Change the default behaviour for prefetching ``--train.prefech 128``.
* Verify that the size of a sample is as small as possible. Python required to pickle the data for parallelization which can drastically slow down the queue-speed.
  We observed crucial problems if the input data size is in the order of more than 50 MB. Consider changing the data type (e.g. ``uint8`` instead of ``int32``)


Optimizing the model
~~~~~~~~~~~~~~~~~~~~

The standard way to increase the throughput of a model is to increase its batch size if the memory of a GPU is not exceeded: ``--train.batch_size 32``.
