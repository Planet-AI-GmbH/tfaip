from dataclasses import dataclass

from paiargparse import pai_dataclass

from examples.imageclassification.params import Keys
from tfaip import Sample, TARGETS_PROCESSOR, INPUT_PROCESSOR
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class PrepareSampleProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return PrepareSampleProcessor


class PrepareSampleProcessor(MappingDataProcessor[PrepareSampleProcessorParams]):
    def apply(self, sample: Sample) -> Sample:
        if self.mode in INPUT_PROCESSOR:
            sample = sample.new_inputs({Keys.Image: sample.inputs})
        if self.mode in TARGETS_PROCESSOR:
            sample = sample.new_targets({Keys.Target: [sample.targets]})
        return sample
