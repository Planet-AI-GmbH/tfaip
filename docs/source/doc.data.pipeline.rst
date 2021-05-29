Data Pipeline
=============

The data pipeline can be used as stand-alone without the requirement to setup a full ``Data`` class.
This can be useful to simply apply a data pipeline comprising a list of processors.

.. code-block:: python

    from tfaip.imports import DataBaseParams

    # Obtain or create data params from a scenario
    # these params also include some
    data_params = scenario.data_params

    # Here run the pre-proc pipeline, but this can be an arbitrary pipeline
    pipeline_params = data_params.pre_proc

    # Create some samples (empty here), can also be a DataGeneratorParams instance
    samples = [Sample()]

    # create the pipeline
    pipeline = pipeline_params.create(DataPipelineParams(num_processes=8, limit=100), data_params)

    # apply the pipeline on the samples and retrieve the output
    for i, d in enumerate(pipeline.apply(samples)):
        print(i, d)
