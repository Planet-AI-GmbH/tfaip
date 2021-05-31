from dataclasses import dataclass

import cv2
from paiargparse import pai_dataclass

from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class RescaleProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return RescaleProcessor


class RescaleProcessor(MappingDataProcessor[RescaleProcessorParams]):
    def apply(self, sample: Sample) -> Sample:
        return sample.new_inputs(
            cv2.resize(sample.inputs, (self.data_params.image_height, self.data_params.image_width))
        )
