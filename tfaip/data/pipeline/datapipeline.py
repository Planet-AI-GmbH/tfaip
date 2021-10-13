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
"""Definition of the base DataPipeline, and RawDataPipeline"""
import copy
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, TypeVar, Generic, Iterable, ContextManager, Tuple, Generator

from tfaip import DataGeneratorParams
from tfaip import Sample, PipelineMode
from tfaip.data.databaseparams import DataPipelineParams
from tfaip.data.pipeline.datagenerator import DataGenerator, RawDataGenerator
from tfaip.data.pipeline.processor.dataprocessor import MappingDataProcessor
from tfaip.data.pipeline.processor.params import DataProcessorPipelineParams, SequentialProcessorPipelineParams
from tfaip.data.pipeline.processor.pipeline import DataProcessorPipeline
from tfaip.data.pipeline.tfdatasetgenerator import TFDatasetGenerator
from tfaip.util.multiprocessing.join import JoinableHolder
from tfaip.util.multiprocessing.parallelmap import tqdm_wrapper

if TYPE_CHECKING:
    from tfaip.data.data import DataBase

logger = logging.getLogger(__name__)

TData = TypeVar("TData", bound="DataBase")


class DataPipeline(Generic[TData], JoinableHolder):
    """
    The DataPipeline sets up and handles the pre- and post-processing pipelines.
    """

    def __init__(
        self,
        pipeline_params: DataPipelineParams,
        data_base: TData,
        generator_params: DataGeneratorParams,
        input_processors: Optional[DataProcessorPipelineParams] = None,
        output_processors: Optional[DataProcessorPipelineParams] = None,
    ):
        super().__init__()
        self.generator_params = generator_params
        self.data = data_base
        self.data_params = data_base.params
        self.mode = pipeline_params.mode
        self.pipeline_params: DataPipelineParams = pipeline_params
        self._auto_batch = True  # Override if batching of the tf.data.Dataset should not be performed automatically
        self._input_processors: DataProcessorPipelineParams = input_processors or self.data_params.pre_proc
        self._output_processors: DataProcessorPipelineParams = output_processors or self.data_params.post_proc

    def to_mode(self, mode: PipelineMode) -> "DataPipeline":
        """
        Clone the pipeline but change its mode
        """
        pipeline_params = deepcopy(self.pipeline_params)
        pipeline_params.mode = mode
        return self.__class__(
            pipeline_params, self.data, self.generator_params, self._input_processors, self._output_processors
        )

    def as_preloaded(self, progress_bar=True) -> "RawDataPipeline":
        """
        Preload all samples of the pipeline (as far as possible) and obtain a new DataPipeline (a `RawDataPipeline`).
        this function is handy if all samples fit in the memory. Further loading and preprocessing of the Samples
        is then not required afterwards.
        """
        if not isinstance(self._input_processors, SequentialProcessorPipelineParams):
            raise TypeError("Preloading is currently only supported for a SequentialProcessorPipeline")
        logger.info(f"Preloading: Converting {self.mode.value} to raw pipeline.")

        # Get the running data pipeline, load the samples, and create a RawDataPipeline
        non_preloadable_params = []
        data = self.preload_input_samples(progress_bar=progress_bar, non_preloadable_params=non_preloadable_params)
        pipeline = RawDataPipeline(data, self.pipeline_params, self.data, self.generator_params)
        pipeline._input_processors = copy.copy(pipeline._input_processors)  # pylint: disable=protected-access
        pipeline._input_processors.processors = non_preloadable_params  # pylint: disable=protected-access
        return pipeline

    @property
    def auto_batch(self):
        return self._auto_batch

    def create_data_generator(self) -> DataGenerator:
        return self.generator_params.create(self.mode)

    def flat_input_processors(self, preload=False, non_preloadable_params=None) -> List[MappingDataProcessor]:
        if not isinstance(self._input_processors, SequentialProcessorPipelineParams):
            raise TypeError(
                "Retrieving of flat input processors is currently only supported for a SequentialProcessorPipeline"
            )
        if non_preloadable_params is None:
            non_preloadable_params = []
        params: SequentialProcessorPipelineParams = self._input_processors

        if params:
            processor_params = [sp for sp in params.processors if self.mode in sp.modes]
            processors = [sp.create(self.data_params, self.mode) for sp in processor_params]
            out_processors = []
            i = 0
            for i, p in enumerate(processors, start=1):
                if p is None:
                    continue  # processor not available in mode
                if preload and not p.supports_preload():
                    i -= 1
                    break
                out_processors.append(p)
            non_preloadable_params.extend(processor_params[i:])
            processors = out_processors
        else:
            processors = []

        return processors

    def _create_tf_dataset_generator(self) -> TFDatasetGenerator:
        return TFDatasetGenerator(self)

    def create_output_pipeline(self) -> Optional[DataProcessorPipeline]:
        params = self._output_processors
        if params:
            return params.create(self.pipeline_params, self.data_params)
        else:
            return DataProcessorPipeline([])

    def input_dataset(self, auto_repeat=None):
        """Return the tf.data.Dataset as input of the network."""
        dataset, _ = self.input_dataset_with_len(auto_repeat)
        return dataset

    def input_dataset_with_len(self, auto_repeat=None):
        gen = self.generate_input_samples(auto_repeat=auto_repeat)
        return gen.as_dataset(self._create_tf_dataset_generator()), len(gen)

    def generate_input_batches(self, auto_repeat=None) -> ContextManager[Generator[Tuple, None, None]]:
        """Network input batches as iterator

        The data will pass the input pipeline and the tf.data.Dataset so these are the actual samples for training, prediction, ...

        Dependent on the mode returns either a tuple (inputs, targets, meta), (inputs, meta), or (targets, meta).

        Usage:
            ```
            with pipeline.generate_input_batches() as batches:
                for batch in batches:
                    print(batch)
        """

        class ContextWrapper:
            def __init__(self, dataset):
                self.dataset = dataset

            def __enter__(self):
                def generator():
                    # This wrapper is required to add "close"-functionality for clean-up
                    batches = self.dataset.as_numpy_iterator()
                    for b in batches:
                        yield b

                self.generator = generator()
                return self.generator

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.generator.close()  # This ensures that upon closing, the pipelines are cleaned up

        return ContextWrapper(self.input_dataset(auto_repeat=auto_repeat))

    def generate_input_samples(self, auto_repeat=None) -> ContextManager[Generator[Sample, None, None]]:
        """Yiels the samples produces by the data generator and after the preproc pipeline.

        Note:
            This function will return a context that must be entered in order to obtain the samples.

            The samples might also be batched already if the DataGenerator if this pipeline yields batches.

        Usage:
            ```
            with pipeline.generate_input_samples() as samples:
                for sample in samples:
                    print(sample)
            ```
        """
        from tfaip.data.pipeline.runningdatapipeline import InputSamplesGenerator

        return InputSamplesGenerator(
            self._input_processors,
            self.data,
            self.create_data_generator(),
            self.pipeline_params,
            self.mode,
            auto_repeat,
        )

    def process_output(self, samples: Iterable[Sample]) -> ContextManager[Generator[Iterable[Sample], None, None]]:
        from tfaip.data.pipeline.runningdatapipeline import OutputSamplesGenerator

        return OutputSamplesGenerator(
            samples,
            self._output_processors,
            self.data,
            self.pipeline_params,
        )

    def preload_input_samples(self, progress_bar=True, non_preloadable_params: Optional[List] = None) -> List[Sample]:
        """
        Applies all pre-proc DataProcessors marked as `preloadable` to the samples of the `DataGenerator` and
        returns a list of the `Samples`. `DataProcessors` that were not applied are added to the
        `non_preloadable_params`.

        See Also:
            `DataPipeline.as_preloaded` uses this to convert the Pipeline to a pipeline with a `RawDataGenerator`
        """
        non_preloadable_params = non_preloadable_params if non_preloadable_params is not None else []
        data_generator = self.create_data_generator()
        old_limit = self.pipeline_params.limit
        self.pipeline_params.limit = len(data_generator)

        # Load the samples
        last_generator = list(
            tqdm_wrapper(
                data_generator.generate(), progress_bar=progress_bar, total=len(data_generator), desc="Loading samples"
            )
        )
        # Obtain the list of preprocessors that are valid to use (preloadable) and apply them one by one
        processors = self.flat_input_processors(preload=True, non_preloadable_params=non_preloadable_params)
        for processor in processors:
            last_generator = processor.preload(
                last_generator,
                num_processes=self.pipeline_params.num_processes,
                progress_bar=progress_bar,
            )
            last_generator = list(last_generator)

        self.pipeline_params.limit = old_limit
        return last_generator


class RawDataPipeline(DataPipeline):
    """
    Implementation of a RawDataPipeline that is directly created with raw input samples
    """

    def __init__(
        self,
        samples: List[Sample],
        pipeline_params: DataPipelineParams,
        data_base: "DataBase",
        generator_params: "DataGeneratorParams",
        input_processors=None,
        output_processors=None,
    ):
        super().__init__(pipeline_params, data_base, generator_params, input_processors, output_processors)
        self.samples = samples

    def to_mode(self, mode: PipelineMode) -> "DataPipeline":
        pipeline_params = copy.deepcopy(self.pipeline_params)
        pipeline_params.mode = mode
        return self.__class__(
            self.samples,
            pipeline_params,
            self.data,
            self.generator_params,
            self._input_processors,
            self._output_processors,
        )

    def create_data_generator(self) -> DataGenerator:
        return RawDataGenerator(self.samples, self.mode, self.generator_params)
