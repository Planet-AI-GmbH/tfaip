from dataclasses import dataclass
from typing import Type

from paiargparse import pai_dataclass
from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor

from examples.atr.params import Keys


@pai_dataclass
@dataclass
class PrepareProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["MappingDataProcessor"]:
        return PrepareProcessor


class PrepareProcessor(MappingDataProcessor):
    def apply(self, sample: Sample) -> Sample:
        img = sample.inputs.transpose()
        encoded = [self.data_params.codec.index(c) for c in sample.targets]
        return sample.new_inputs({Keys.Image: img, Keys.ImageLength: [img.shape[0]]}).new_targets(
            {Keys.Targets: encoded, Keys.TargetsLength: [len(encoded)]}
        )
