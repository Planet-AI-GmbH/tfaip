from abc import ABC, abstractmethod
from random import shuffle
from typing import Iterable, List

from tfaip.base import DataGeneratorParams
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample


class DataGenerator(ABC):
    def __init__(self, mode: PipelineMode, params: 'DataGeneratorParams'):
        params.validate()
        self.mode = mode
        self.params = params

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self) -> Iterable[Sample]:
        raise NotImplementedError


class RawDataGenerator(DataGenerator):
    def __init__(self, raw_data: List[Sample], mode: PipelineMode, params: 'DataGeneratorParams'):
        super(RawDataGenerator, self).__init__(mode, params)
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def generate(self) -> Iterable[Sample]:
        if self.mode == PipelineMode.Training:
            shuffle(self.raw_data)
        return self.raw_data
