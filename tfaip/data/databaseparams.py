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
"""Definition of DataBaseParams, DataPipelineParams, and DataGeneratorParams"""
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type, List, Optional

from paiargparse import pai_meta, pai_dataclass

from tfaip.data.pipeline.definitions import PipelineMode
from tfaip.data.pipeline.processor.params import (
    SequentialProcessorPipelineParams,
    ComposedProcessorPipelineParams,
    DataProcessorPipelineParams,
)

if TYPE_CHECKING:
    from tfaip.data.pipeline.datagenerator import DataGenerator
    from tfaip.data.data import DataBase

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class DataGeneratorParams(ABC):
    """
    Parameter class that defines how to construct a DataGenerator.
    """

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        raise NotImplementedError

    def create(self, mode: PipelineMode) -> "DataGenerator":
        return self.cls()(mode, self)


@pai_dataclass
@dataclass
class DataPipelineParams:
    """
    Parameter class that defines the general parameters, e.g. batch size, prefetching, number of processes, ...
    of a certain (e.g. train or val) pipeline.
    """

    batch_size: int = field(default=16, metadata=pai_meta(help="Batch size"))
    limit: int = field(
        default=-1,
        metadata=pai_meta(
            help="Limit the number of examples produced by the generator. Note, if GeneratingDataProcessors are present "
            "in the data pipeline, the number of examples produced by the generator can differ."
        ),
    )
    prefetch: int = field(
        default=-1,
        metadata=pai_meta(help="Prefetching data. -1 default to max(num_processes * 2 by default, 2 * batch size)"),
    )
    num_processes: int = field(default=4, metadata=pai_meta(help="Number of processes for data loading."))
    batch_drop_remainder: bool = field(
        default=False,
        metadata=pai_meta(
            help="Drop remainder parameter of padded_batch. Drop batch if it is smaller than batch size."
        ),
    )
    shuffle_buffer_size: int = field(
        default=-1,
        metadata=pai_meta(
            help="Size of the shuffle buffer required for randomizing data (if required). Disabled by default."
        ),
    )
    mode: PipelineMode = field(
        default=PipelineMode.TRAINING,
        metadata=pai_meta(
            mode="ignore", help="Mode of this pipeline. To be set automatically by modules (train, predict, lav...)"
        ),
    )
    bucket_boundaries: List[int] = field(
        default_factory=list,
        metadata=pai_meta(
            help="Elements of the Dataset are grouped together by length and then are padded and batched. "
            "See tf.data.experimental.bucket_by_sequence_length"
        ),
    )
    bucket_batch_sizes: Optional[List[int]] = field(
        default=None,
        metadata=pai_meta(help="Batch sizes of the buckets. By default, batch_size * (len(bucked_boundaries) + 1)."),
    )

    def __post_init__(self):
        if self.num_processes <= 0:
            raise ValueError(f"Number of processes must be > 0 but got {self.num_processes}")

        if self.prefetch < 0:
            self.prefetch = 8 * max(self.num_processes, self.batch_size // self.num_processes)


@pai_dataclass
@dataclass
class DataBaseParams(ABC):
    """
    Parameters that define the overall setup of the data pipelines (pre_proc and post_proc)

    Parameters of this class will be shared among all DataProcessors.
    """

    @staticmethod
    @abstractmethod
    def cls() -> Type["DataBase"]:
        raise NotImplementedError

    def create(self) -> "DataBase":
        return self.cls()(params=self)

    # Store pre- and post-processing
    pre_proc: DataProcessorPipelineParams = field(
        default_factory=SequentialProcessorPipelineParams,
        metadata=pai_meta(choices=[SequentialProcessorPipelineParams, ComposedProcessorPipelineParams]),
    )
    post_proc: DataProcessorPipelineParams = field(
        default_factory=SequentialProcessorPipelineParams,
        metadata=pai_meta(choices=[SequentialProcessorPipelineParams, ComposedProcessorPipelineParams]),
    )

    # Other params
    resource_base_path: str = field(
        default=os.getcwd(), metadata=pai_meta(mode="ignore", help="Path where to find the resources")
    )
