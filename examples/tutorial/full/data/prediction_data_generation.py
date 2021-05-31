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
import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from paiargparse import pai_meta, pai_dataclass

from tfaip import DataGeneratorParams
from tfaip import PipelineMode, Sample


def to_samples(samples):
    return [
        Sample(inputs={"img": img}, targets={"gt": gt.reshape((1,))}, meta={"index": i})
        for i, (img, gt) in enumerate(zip(*samples))
    ]


@pai_dataclass
@dataclass
class TutorialPredictionGeneratorParams(DataGeneratorParams):
    """
    Parameters for a data generator that loads images (and optional their gt) for prediction and LAV

    GT is expected to be written in a plain text file that has the same name as the corresponding image with a ".txt"
    suffix.
    """

    files: str = field(
        default="",
        metadata=pai_meta(
            required=True, help="Path to the image to load. Use wildcards '*' to provide multiple files."
        ),
    )

    @staticmethod
    def cls():
        raise NotImplementedError

    def create(self, mode: PipelineMode):
        # Here, all samples are loaded and passed to a RawDataGenerator
        from tfaip.data.pipeline.datagenerator import RawDataGenerator

        assert self.files, "No images provided"

        def load_sample(fn) -> Sample:
            img = cv2.imread(fn, flags=cv2.IMREAD_GRAYSCALE)
            gt_path = fn + ".txt"
            if os.path.exists(gt_path):
                with open(gt_path) as f:
                    gt = np.asarray([int(f.read())])
            else:
                gt = None
            return Sample(inputs={"img": img}, targets={"gt": gt}, meta={"fn": fn})

        return RawDataGenerator(list(map(load_sample, glob.glob(self.files))), mode, self)
