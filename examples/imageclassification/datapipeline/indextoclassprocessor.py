from dataclasses import dataclass

from paiargparse import pai_dataclass

from examples.imageclassification.params import Keys
from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class IndexToClassProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return IndexToClassProcessor


class IndexToClassProcessor(MappingDataProcessor):
    def apply(self, sample: Sample) -> Sample:
        sample.outputs[Keys.OutputClassName] = self.data_params.classes[sample.outputs[Keys.OutputClass]]
        return sample
