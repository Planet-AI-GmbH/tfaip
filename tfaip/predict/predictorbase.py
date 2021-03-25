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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, Iterable, Union, Optional, Any

import prettytable
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import tf_utils

from tfaip.data.databaseparams import DataGeneratorParams, DataPipelineParams
from tfaip import PipelineMode, Sample
from tfaip import PredictorParams
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.data.pipeline.datapipeline import DataPipeline
from tfaip.device.device_config import DeviceConfig, distribute_strategy
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper
from tfaip.util.profiling import MeasureTime
from tfaip.util.tfaipargparse import post_init

if TYPE_CHECKING:
    from tfaip.data.data import DataBase

logger = logging.getLogger(__name__)


@dataclass
class PredictorBenchmarkResults:
    """Class storing the benchmark of a full prediction

    Use pretty_print to print a formatted table of the full results.
    """
    n_batches: float = 0
    n_samples: float = 0
    total_time: float = 1e-10
    avg_time_per_batch: float = 0
    avg_time_per_sample: float = 0
    batches_per_second: float = 0
    samples_per_second: float = 0

    def pretty_print(self):
        table = prettytable.PrettyTable(['', 'Total', 'Batch', 'Sample'])
        table.add_row(['Count', 1, self.n_batches, self.n_samples])
        table.add_row(['Time Per', self.total_time, self.avg_time_per_batch, self.avg_time_per_sample])
        table.add_row(['Per Second', 1 / self.total_time, self.batches_per_second, self.samples_per_second])
        print(table)


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

    def __init__(self, params: PredictorParams, data: 'DataBase'):
        post_init(params)
        self._params = params
        self._params.pipeline.mode = PipelineMode.PREDICTION
        if params.include_targets:
            self._params.pipeline.mode = PipelineMode.EVALUATION
        self.device_config = DeviceConfig(self._params.device)
        self._data = data
        self.benchmark_results = PredictorBenchmarkResults()
        self._keras_model: Optional[keras.Model] = None

    @property
    def params(self):
        return self._params

    @property
    def data(self):
        return self._data

    def _load_model(self, model: Union[str, keras.Model], convert_to_input_output=True):
        if isinstance(model, str):
            model = keras.models.load_model(model, compile=False)

        # wrap to output inputs as outputs
        if convert_to_input_output:
            inputs = self._data.create_input_layers()
            meta = self._data.create_meta_as_input_layers()
            outputs = model(inputs)
            if self._params.include_targets:
                targets = self._data.create_target_as_input_layers()
                joined = {**inputs, **targets, **meta}
                model = keras.models.Model(inputs=joined, outputs=(inputs, targets, outputs, meta))
            else:
                joined = {**inputs, **meta}
                model = keras.models.Model(inputs=joined, outputs=(inputs, outputs, meta))

        model.run_eagerly = self._params.run_eagerly
        return model

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
        """
        if size is None:
            # Automatically compute the size (number of samples)
            try:
                size = len(inputs)
            except TypeError:
                logger.warning(
                    'Size not specified. Converting inputs to list to obtain size. Or implement size on inputs')
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
            DataPipelineParams(mode=PipelineMode.PREDICTION),
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
                    desc='Prediction',
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

        # join all input data in one dict, which is passed "around" the actual model an split up afterwards
        if self._params.include_targets:
            dataset = dataset.map(lambda i, t, m: {**i, **t, **m})
        else:
            dataset = dataset.map(lambda i, m: {**i, **m})

        if self._keras_model is None:
            raise ValueError('No model set. Call predictor.set_model(model)')
        with MeasureTime() as total_time:
            # The following code is copied from keras.model.Model.predict
            # It sets up the distribution strategy, the DataSet (here DataHandler)
            # Then one epoch is iterated until catch_stop_iteration() is reached
            with self._keras_model.distribute_strategy.scope():
                data_handler = data_adapter.DataHandler(dataset)
                predict_function = self._keras_model.make_predict_function()
                for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
                    with data_handler.catch_stop_iteration():
                        for _ in data_handler.steps():
                            with MeasureTime() as batch_time:
                                r = predict_function(iterator)  # hack to access inputs

                                # If targets are included, the return value differs
                                if self._params.include_targets:
                                    inputs, targets, outputs, meta = tf_utils.to_numpy_or_python_type(r)
                                else:
                                    inputs, outputs, meta = tf_utils.to_numpy_or_python_type(r)
                                    targets = {}  # No targets in normal prediction

                                # split into single samples
                                try:
                                    batch_size = next(iter(inputs.values())).shape[0]
                                except StopIteration as e:
                                    raise ValueError(f'Empty inputs {inputs}. This should never occur!') from e
                                for sample in self._unwrap_batch(inputs, targets, outputs, meta):
                                    self._on_sample_end(sample)
                                    yield sample

                            # Some Benchmarks
                            self.benchmark_results.n_batches += 1
                            self.benchmark_results.n_samples += batch_size
                            self.benchmark_results.avg_time_per_batch += batch_time.duration
                            self.benchmark_results.avg_time_per_sample += batch_time.duration
                            self._on_step_end(Sample(inputs=inputs, outputs=outputs, targets=targets, meta=meta))

        # Overall Benchmarks
        self.benchmark_results.total_time = total_time.duration
        self.benchmark_results.avg_time_per_batch /= self.benchmark_results.n_batches
        self.benchmark_results.avg_time_per_sample /= self.benchmark_results.n_samples
        self.benchmark_results.batches_per_second = 1 / self.benchmark_results.avg_time_per_batch
        self.benchmark_results.samples_per_second = 1 / self.benchmark_results.avg_time_per_sample

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
