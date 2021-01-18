# Copyright 2020 The tfaip authors. All Rights Reserved.
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
from dataclasses import dataclass, field
import logging
import os
from typing import Optional

from dataclasses_json import dataclass_json

from tfaip.base.data.pipeline.sample.params import SamplePipelineParams
from tfaip.util.argumentparser import dc_meta

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class DataGeneratorParams:
    batch_size: int = field(default=16, metadata=dc_meta(
        help="Batch size"
    ))
    limit: int = field(default=-1, metadata=dc_meta(
        help="Limit the number of examples"
    ))
    prefetch: int = field(default=-1, metadata=dc_meta(
        help="Prefetching data. -1 is max(num_processes * 2 by default, 2 * batch size)"
    ))
    num_processes: int = field(default=4, metadata=dc_meta(
        help="Number of processes for data loading."
    ))
    batch_drop_remainder: bool = field(default=False, metadata=dc_meta(
        help="Drop remainder parameter of padded_batch. Drop batch if it is smaller than batch size."
    ))

    def validate(self):
        if self.num_processes <= 0:
            raise ValueError(f"Number of processes must be > 0 but got {self.num_processes}")

        if self.prefetch < 0:
            self.prefetch = 8 * max(self.num_processes, self.batch_size // self.num_processes)


@dataclass_json
@dataclass
class DataBaseParams:
    # override those files in derived params to set the correct pipeline params
    train: DataGeneratorParams = field(default_factory=DataGeneratorParams, metadata=dc_meta(arg_mode='snake'))
    val: DataGeneratorParams = field(default_factory=DataGeneratorParams, metadata=dc_meta(arg_mode='snake'))

    # Processing params
    preproc_max_tasks_per_child: int = field(default=250, metadata=dc_meta(
        help="Maximum tasks of a child in the preproc pipeline after a child is recreated. "
             "Higher numbers for better performance but on the drawback if higher memory consumption. "
             "Only used if the scenario uses a DataPipeline."
    ))

    # Store pre- and post-processing
    pre_processors_: Optional[SamplePipelineParams] = None
    post_processors_: Optional[SamplePipelineParams] = None

    # Other params
    resource_base_path_: str = os.getcwd()
