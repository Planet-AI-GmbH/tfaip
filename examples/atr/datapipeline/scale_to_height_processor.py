from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
from paiargparse import pai_dataclass
from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class ScaleToHeightProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return ScaleToHeightProcessor


class ScaleToHeightProcessor(MappingDataProcessor[ScaleToHeightProcessorParams]):
    def apply(self, sample: Sample) -> Sample:
        assert self.data_params.height > 0  # Not initialized
        return sample.new_inputs(scale_to_h(sample.inputs, self.data_params.height))


def scale_to_h(img, target_height):
    assert img.dtype == np.uint8

    h, w = img.shape[:2]
    if h == target_height:
        return img
    if h == 0 or img.size == 0:
        # empty image
        return np.zeros(shape=(target_height, w) + img.shape[2:], dtype=img.dtype)

    scale = target_height * 1.0 / h
    target_width = np.maximum(round(scale * w), 1)
    if scale <= 1:
        # Down-Sampling: interpolation "area"
        return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_AREA)

    else:
        # Up-Sampling: linear interpolation
        return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)
