# Copyright 2021 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
"""Definition of the PredictorBase"""
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, Iterable, Union, Optional, Any

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

from tfaip.data.databaseparams import DataGeneratorParams, DataPipelineParams
from tfaip import PipelineMode, Sample
from tfaip import PredictorParams
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.data.pipeline.datapipeline import DataPipeline
from tfaip.device.device_config import DeviceConfig, distribute_strategy
from tfaip.model.modelbase import ModelBase
from tfaip.predict.raw_predictor import RawPredictor
from tfaip.trainer.callbacks.benchmark_callback import BenchmarkResults
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper
from tfaip.util.profiling import MeasureTime
from tfaip.util.tfaipargparse import post_init
from tfaip.util.tftyping import sync_to_numpy_or_python_type

if TYPE_CHECKING:
    from tfaip.data.data import DataBase

logger = logging.getLogger(__name__)


class PredictorBase(ABC):
    """
    The PredictorBase handles the prediction of data.
    For instantiation or subclassing see Predictor or MultiPredictor.


    Several data formats are allowed:
    - predict: predict with given DataGeneratorParams
    - predict_raw: predict with a given raw sample
    - predict_pipeline: predict a DataPipeline
    - predict_dataset: predict a tf.data.Dataset
    predict and predict_raw will automatically apply the pre_proc pipeline of Data.
    The post_proc pipeline is always applied.

    Metrics (e.g. timing) of the prediction will be stored in benchmark_results
    See Also:
        - Predictor
        - MultiPredictor
        - ScenarioBase.create_predictor
        - ScenarioBase.create_multi_predictor
        - LAV

    """

    @classmethod
    def params_cls(cls) -> Type[PredictorParams]:
        return PredictorParams

    def __init__(self, params: PredictorParams, data: "DataBase"):
        post_init(params)
        self._params = params
        self._params.pipeline.mode = PipelineMode.PREDICTION
        if params.include_targets:
            self._params.pipeline.mode = PipelineMode.EVALUATION
        self.device_config = DeviceConfig(self._params.device)
        self._data = data
        self.benchmark_results = BenchmarkResults()
        self._keras_model: Optional[keras.Model] = None

    @property
    def params(self):
        return self._params

    @property
    def data(self):
        return self._data

    def _load_model(self, model: Union[str, keras.Model]):
        if isinstance(model, str):
            model = keras.models.load_model(model, compile=False, custom_objects=ModelBase.base_custom_objects())

        return model

    def raw(self) -> RawPredictor:
        """Create a raw predictor from this predictor that allows to asynchronly predict raw samples.

        Usage:

        Either call

        with predictor.raw() as raw_pred:
            raw_pred(sample1)
            raw_pred(sample2)
            ...

        or

        raw_pred = predictor.raw().__enter__()
        raw_pred(sample1)
        raw_pred(sample2)

        """
        return RawPredictor(self)

    def predict(self, params: DataGeneratorParams) -> Iterable[Sample]:
        """
        Predict a DataGenerator based on its params.
        The generated samples will be fed in the pre_proc pipeline, the model and the post_proc pipeline

        Args:
            params: DataGeneratorParams

        Returns:
            The predicted Samples
        """
        return self.predict_pipeline(self._data.create_pipeline(self.params.pipeline, params))

    @abstractmethod
    def _unwrap_batch(self, inputs, targets, outputs, meta) -> Iterable[Sample]:
        """
        Implemented in Predictor and MultiPredictor since the actual model defines how to split a batch.
        """
        raise NotImplementedError

    def predict_raw(self, inputs: Iterable[Any], *, size=None) -> Iterable[Sample]:
        """
        Predict input samples.

        The raw samples will directly be fed in the pre_proc pipeline, the model and the post_proc pipeline
        Args:
            inputs: Raw input which must match to the expected input of the pre_proc pipeline of Data
            size: The number of samples (to be expected).
                  If None, the inputs will be converted to a list to automatically compute the size.
                  Hint, but also Warning: Set this to 1 if the length is unknown. In this case however the progress
                  cant be computed.

        Returns:
            The predicted samples

        See Also:
            - predict_pipeline
            - predict_dataset
            - predict
            - raw
        """
        if size is None:
            # Automatically compute the size (number of samples)
            try:
                size = len(inputs)
            except TypeError:
                logger.warning(
                    "Size not specified. Converting inputs to list to obtain size. Or implement size on inputs"
                )
                inputs = list(inputs)
            size = len(inputs)

        # Setup a pipeline using the daw data generator as Input
        class RawGenerator(DataGenerator):
            def __len__(self):
                return size

            def generate(self) -> Iterable[Sample]:
                return map(lambda x: Sample(inputs=x, meta={}), inputs)

        class RawInputsPipeline(DataPipeline):
            def create_data_generator(self) -> DataGenerator:
                return RawGenerator(mode=self.mode, params=self.generator_params)

        pipeline = RawInputsPipeline(
            self.params.pipeline,
            self._data,
            DataGeneratorParams(),
        )

        return self.predict_pipeline(pipeline)

    def predict_pipeline(self, pipeline: DataPipeline) -> Iterable[Sample]:
        """
        Predict a instantiated DataPipeline which is used for data generation, pre- and post-processing.

        Args:
            pipeline: the DataPipeline

        Returns:
            The preprocessed samples

        See Also:
            - predict_dataset
            - predict_raw
            - predict
        """
        with pipeline as rd:
            for sample in tqdm_wrapper(
                rd.process_output(self.predict_dataset(rd.input_dataset())),
                progress_bar=self._params.progress_bar,
                desc="Prediction",
                total=len(rd),
            ):
                if not self._params.silent:
                    self._print_prediction(sample, logger.info)

                yield sample

    @distribute_strategy
    def predict_dataset(self, dataset: tf.data.Dataset) -> Iterable[Sample]:
        """
        Apply the prediction model on the tf.data.Dataset. No pre- or post-processing will be applied.

        Args:
            dataset: The tf.data.Dataset with one dictionary as output or two if PredictionParams.include_targets = True

        Returns:
            The raw predicted Samples of the model

        See Also:
            - predict
            - predict_pipeline
            - predict_raw
        """
        if self._keras_model is None:
            raise ValueError("No model set. Call predictor.set_model(model)")

        keras_model = self._keras_model

        class WrappedModel(tf.keras.models.Model):
            def __init__(self, with_targets, **kwargs):
                super().__init__(**kwargs)
                self.with_targets = with_targets

            def call(self, inputs, training=None, mask=None):
                if self.with_targets:
                    inputs, targets, meta = inputs
                    return inputs, targets, keras_model(inputs), meta
                else:
                    inputs, meta = inputs
                    return inputs, keras_model(inputs), meta

            def get_config(self):
                raise NotImplementedError

        # wrap model so that it outputs inputs, meta and optionally the targets
        wrapped_model = WrappedModel(self.params.include_targets)
        wrapped_model.compile(run_eagerly=self.params.run_eagerly)

        if self._params.include_targets:
            dataset = dataset.map(lambda i, t, m: ((i, t, m),))
        else:
            dataset = dataset.map(lambda i, m: ((i, m),))

        with MeasureTime() as total_time:
            # The following code is copied from keras.model.Model.predict
            # It sets up the distribution strategy, the DataSet (here DataHandler)
            # Then one epoch is iterated until catch_stop_iteration() is reached
            with self._keras_model.distribute_strategy.scope():
                data_handler = data_adapter.DataHandler(dataset)
                predict_function = wrapped_model.make_predict_function()
                for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
                    with data_handler.catch_stop_iteration():
                        for _ in data_handler.steps():
                            with MeasureTime() as batch_time:
                                r = predict_function(iterator)  # hack to access inputs

                                # If targets are included, the return value differs
                                if self._params.include_targets:
                                    inputs, targets, outputs, meta = sync_to_numpy_or_python_type(r)
                                else:
                                    inputs, outputs, meta = sync_to_numpy_or_python_type(r)
                                    targets = {}  # No targets in normal prediction

                                # split into single samples
                                try:
                                    batch_size = tf.nest.flatten(inputs)[0].shape[0]
                                except StopIteration as e:
                                    raise ValueError(f"Empty inputs {inputs}. This should never occur!") from e
                                for sample in self._unwrap_batch(inputs, targets, outputs, meta):
                                    self._on_sample_end(sample)
                                    yield sample

                            # Some Benchmarks
                            self.benchmark_results.finish_batch(batch_size, batch_time.duration)
                            self._on_step_end(Sample(inputs=inputs, outputs=outputs, targets=targets, meta=meta))

        # Overall Benchmarks
        self.benchmark_results.finish_epoch(total_time.duration)
        self._on_predict_end()

    @abstractmethod
    def _print_prediction(self, sample: Sample, print_fn):
        raise NotImplementedError

    def _on_sample_end(self, outputs):
        pass

    def _on_step_end(self, outputs):
        pass

    def _on_predict_end(self):
        pass
