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
from abc import ABC, abstractmethod
from typing import List, Type, Optional, Tuple, Any, Iterable, Iterator, Dict
import logging

from typeguard import typechecked

from tfaip.base.data.pipeline.definitions import DataProcessorFactoryParams, INPUT_PROCESSOR, \
    TARGETS_PROCESSOR, Sample, PipelineMode
from tfaip.util.multiprocessing.parallelmap import parallel_map


logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    @staticmethod
    def default_params() -> dict:
        return {}

    def __init__(self,
                 params,
                 mode: PipelineMode,
                 ):
        super().__init__()
        self.params = params
        self.mode = mode

    def supports_preload(self):
        return True

    @typechecked
    def __call__(self, sample: Sample) -> Sample:
        return self.apply(sample)

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        raise NotImplementedError

    def preload(self,
                samples: List[Sample],
                num_processes=1,
                drop_invalid=True,
                progress_bar=False,
                ) -> Iterable[Sample]:
        return self.apply_on_samples(samples, num_processes, drop_invalid, progress_bar)

    def apply_on_samples(self,
                         samples: Iterable[Sample],
                         num_processes=1,
                         drop_invalid=True,
                         progress_bar=False,
                         ) -> Iterator[Sample]:
        mapped = parallel_map(self.apply_on_sample, samples, processes=num_processes, progress_bar=progress_bar,
                              desc=f"Applying data processor {self.__class__.__name__}"
                              )
        if drop_invalid:
            mapped = filter(lambda s: s is not None, mapped)
        return mapped

    def apply_on_sample(self, sample: Sample) -> Sample:
        if sample.meta is None:
            sample = sample.new_meta({})
        return self.apply(sample.copy())


class SequenceProcessor(DataProcessor):
    def split_by_index(self, index) -> Tuple['SequenceProcessor', 'SequenceProcessor']:
        return (SequenceProcessor(self.params, self.mode, self.processors[:index]),
                SequenceProcessor(self.params, self.mode, self.processors[index:]))

    def split_by_processor(self, p: Type[DataProcessor]) -> Tuple['SequenceProcessor', 'SequenceProcessor']:
        i = 0
        for i, x in self.processors:
            if type(x) == p:
                break
        return self.split_by_processor(i)

    def is_valid_sample(self, sample: Sample) -> bool:
        if sample.inputs is None and self.mode in INPUT_PROCESSOR:
            return False
        if sample.targets is None and self.mode in TARGETS_PROCESSOR:
            return False
        return True

    def apply(self, sample: Sample) -> Optional[Sample]:
        if sample.meta is None:
            sample = sample.new_meta({})

        if not self.is_valid_sample(sample):
            return None

        for p in self.processors:
            sample = p(sample)
            if not self.is_valid_sample(sample):
                return None

        return sample

    def __init__(self, params, mode, processors: List[DataProcessor]):
        super(SequenceProcessor, self).__init__(params, mode)
        self.processors = processors


class DataProcessorFactory:
    def __init__(self,
                 processors: List[Type[DataProcessor]]
                 ):
        self.processors: Dict[str, Type[DataProcessor]] = {cls.__name__: cls for cls in processors}

    def create(self, factory_params: DataProcessorFactoryParams, params, mode) -> Optional[DataProcessor]:
        if mode in factory_params.modes:
            cls: Type[DataProcessor] = self.processors[factory_params.name]
            args = factory_params.args if factory_params.args is not None else cls.default_params()
            return cls(params=params, mode=mode, **args)

        logger.debug(f"Ignoring {factory_params.name} since the pipeline mode {mode} is not in its modes {[m.value for m in factory_params.modes]}")
        return None

    def create_sequence(self, factory_params: List[DataProcessorFactoryParams], params, mode) -> SequenceProcessor:
        return SequenceProcessor(params, mode, list(filter(lambda x: x, [self.create(p, params, mode) for p in factory_params])))
