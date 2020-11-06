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
from typing import Optional, List
import logging
import os

from dataclasses_json import dataclass_json

from tfaip.util.argument_parser import dc_meta

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class DataBaseParams:
    val_list: str = field(default=None, metadata=dc_meta(
        help="The validation list for testing the model during training."
    ))
    val_batch_size: int = field(default=16, metadata=dc_meta(
        help="The batch size during validation or LAV"
    ))
    val_limit: int = field(default=-1, metadata=dc_meta(
        help="Limit the number of validation examples"
    ))
    val_prefetch: int = field(default=-1, metadata=dc_meta(
        help="Prefetching validation data. -1 is max(val_num_processes * 2 by default, 2 * batch size)"
    ))
    val_num_processes: int = field(default=-1, metadata=dc_meta(
        help="Number of processes for validation data loading. Is train_num_processes by default."
             "Must be supported by the scenario."
    ))
    val_batch_drop_remainder: bool = field(default=False, metadata=dc_meta(
        help="Drop remainder parameter of padded_batch. Drop batch if it is smaller than batch size."
    ))

    train_lists: List[str] = field(default=None, metadata=dc_meta(
        help="Training list files."
    ))
    train_list_ratios: List[float] = field(default=None, metadata=dc_meta(
        help="Ratios of picking list files. Must be supported by the scenario"
    ))
    train_batch_size: int = field(default=16, metadata=dc_meta(
        help="Batch size during training. "
             "To allow for bigger batch sizes --trainer_params train_accum_steps can be used"
    ))
    train_limit: int = field(default=-1, metadata=dc_meta(
        help="Use this for debugging to overfit on a portion of the same data."
    ))
    train_prefetch: int = field(default=-1, metadata=dc_meta(
        help="Prefetch of training examples. Default is max(train_num_processes * 8, 8 * batch size)"
    ))
    train_num_processes: int = field(default=4, metadata=dc_meta(
        help="Number of processes in the training dataset (must be supported by the scenario)"
    ))
    train_batch_drop_remainder: bool = field(default=False, metadata=dc_meta(
        help="Drop remainder parameter of padded_batch. Drop batch if it is smaller than batch size."
    ))

    lav_lists: Optional[List[str]] = field(default=None, metadata=dc_meta(
        help="List to use for LAV (to enable LAV during training set --trainer_params lav_every_n). By default use the"
             "validation list."
    ))

    preproc_max_tasks_per_child: int = field(default=250, metadata=dc_meta(
        help="Maximum tasks of a child in the preproc pipeline after a child is recreated. "
             "Higher numbers for better performance but on the drawback if higher memory consumption. "
             "Only used if the scenario uses a DataPipeline."
    ))

    resource_base_path_: str = os.getcwd()

    def validate(self):
        if self.train_num_processes <= 0:
            raise ValueError(f"Number of processes must be > 0 but got {self.train_num_processes}")

        if self.val_num_processes < 0:
            self.val_num_processes = self.train_num_processes

        if self.val_prefetch < 0:
            self.val_prefetch = 8 * max(self.val_num_processes, self.val_batch_size // self.val_num_processes)

        if self.train_prefetch < 0:
            self.train_prefetch = 8 * max(self.train_num_processes, self.train_batch_size // self.train_num_processes)

        if self.lav_lists:
            if not self.val_list:
                self.val_list = self.lav_lists[0]

        if self.train_lists:
            if not self.train_list_ratios:
                self.train_list_ratios = [1.0] * len(self.train_lists)
            else:
                if len(self.train_list_ratios) != len(self.train_lists):
                    raise ValueError("Length of train_list_ratios must be equals to number of train_lists. Got {}!={}".format(len(self.train_list_ratios), len(self.train_lists)))

