from dataclasses import dataclass, field
from typing import Optional, List
import logging

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

