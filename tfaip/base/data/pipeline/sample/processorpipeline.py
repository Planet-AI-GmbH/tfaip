from typing import List, Optional, Callable, Iterable, TYPE_CHECKING

from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory, SequenceProcessor
from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, Sample, PipelineMode
from tfaip.base.data.pipeline.parallelpipeline import ParallelDataProcessorPipeline

if TYPE_CHECKING:
    from tfaip.base.data.pipeline.datapipeline import DataPipeline


def create_processor_fn(factory: DataProcessorFactory, processors: List[DataProcessorFactoryParams], params, mode: PipelineMode) -> SequenceProcessor:
    return factory.create_sequence(processors, params, mode)


class SampleProcessorPipeline:
    def __init__(self, data_pipeline: 'DataPipeline', processor_fn: Optional[Callable[[], SequenceProcessor]] = None):
        self.data_pipeline = data_pipeline
        self.create_processor_fn = processor_fn

    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        if not self.create_processor_fn:
            for sample in samples:
                yield sample
        else:
            processor = self.create_processor_fn()
            for sample in samples:
                r = processor.apply_on_sample(sample)
                if r is not None:
                    yield r


class ParallelSampleProcessingPipeline(SampleProcessorPipeline):
    def apply(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        parallel_pipeline = ParallelDataProcessorPipeline(self.data_pipeline, samples,
                                                          create_processor_fn=self.create_processor_fn,
                                                          auto_repeat_input=False)
        for x in parallel_pipeline.output_generator():
            yield x

        parallel_pipeline.join()
