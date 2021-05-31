from dataclasses import dataclass
from typing import Type
import cv2

from paiargparse import pai_dataclass
from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class DecoderProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["MappingDataProcessor"]:
        return DecoderProcessor


class DecoderProcessor(MappingDataProcessor):
    def apply(self, sample: Sample) -> Sample:
        sample.outputs["sentence"] = "".join(self.data_params.codec[i] for i in sample.outputs["decoded"] if i >= 0)
        return sample
