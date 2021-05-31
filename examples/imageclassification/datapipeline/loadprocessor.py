from dataclasses import dataclass

import cv2
from paiargparse import pai_dataclass

from tfaip import Sample, INPUT_PROCESSOR, TARGETS_PROCESSOR
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class LoadProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return LoadProcessor


class LoadProcessor(MappingDataProcessor):
    def apply(self, sample: Sample) -> Sample:
        # Load the image and convert the class name into an index
        if self.mode in INPUT_PROCESSOR:
            sample = sample.new_inputs(cv2.imread(sample.inputs))
        if self.mode in TARGETS_PROCESSOR:
            # Only load target if it is (expected to be) present
            sample = sample.new_targets(self.data_params.classes.index(sample.targets))
        return sample
