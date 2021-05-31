from dataclasses import dataclass
from typing import Type
import cv2

from paiargparse import pai_dataclass
from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class LoadProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["MappingDataProcessor"]:
        return LoadProcessor


class LoadProcessor(MappingDataProcessor):
    def apply(self, sample: Sample) -> Sample:
        img = cv2.imread(sample.inputs, flags=cv2.IMREAD_GRAYSCALE)
        with open(sample.targets) as f:
            txt = f.read().strip()

        return sample.new_inputs(img).new_targets(txt)
