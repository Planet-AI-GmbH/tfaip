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
"""Definition of the DataGenerator"""
from abc import ABC, abstractmethod
from random import shuffle
from typing import Iterable, List, TypeVar, Generic

from tfaip import DataGeneratorParams
from tfaip import PipelineMode, Sample

T = TypeVar("T", bound=DataGeneratorParams)


class DataGenerator(Generic[T], ABC):
    """
    The purpose of a DataGenerator is to generate Samples for a given mode.
    A Generator is constructed by is DataGeneratorParams

    An implemented DataGenerator must implement generate and __len__.
    """

    def __init__(self, mode: PipelineMode, params: T):
        self.mode = mode
        self.params = params

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self) -> Iterable[Sample]:
        raise NotImplementedError


class RawDataGenerator(DataGenerator[T]):
    """
    The RawDataGenerator is a derived data generator that is already initialized with loaded samples which are
    yielded in generate().

    If a RawDataGenerator shall be created by a DataGeneratorParams-class, override DataGeneratorParams.create()
    """

    def __init__(self, raw_data: List[Sample], mode: PipelineMode, params: T):
        super().__init__(mode, params)
        self.raw_data = raw_data
        self.shuffle = mode == PipelineMode.TRAINING

    def __len__(self):
        return len(self.raw_data)

    def generate(self) -> Iterable[Sample]:
        if self.shuffle:
            shuffle(self.raw_data)
        return self.raw_data
