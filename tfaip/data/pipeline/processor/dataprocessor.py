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
"""Definition of the DataProcessor, MappingDataProcessor, GeneratingDataProcessor, and DataProcessorParams

The DataProcessorParams create their corresponding DataProcessor
(either MappingDataProcessor or GeneratingDataProcessor).
"""
import logging
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Iterator, Union, TYPE_CHECKING, Type, Set, TypeVar, Generic

from paiargparse import pai_dataclass, pai_meta
from typeguard import typechecked

from tfaip.data.pipeline.definitions import INPUT_PROCESSOR, TARGETS_PROCESSOR, Sample, PipelineMode, GENERAL_PROCESSOR
from tfaip.util.multiprocessing.parallelmap import parallel_map

if TYPE_CHECKING:
    from tfaip import DataBaseParams

logger = logging.getLogger(__name__)


class Key(str):
    """General keys to be used in the input, target, and output dicts of a DataProcessor or the Model"""

    IMG = "img"
    IMG_SHAPE = "img_shape"
    IMG_PATH = "img_path"
    MASK = "mask"
    GT_BOXES = "groundtruth_boxes"
    GT_CLASSES = "groundtruth_classes"
    GT_NAMES = "groundtruth_names"
    GT_NUM = "groundtruth_boxes_num"
    PRED = "pred"  # prediction which are values in [0,1] for each class-dimension. Last dimension is class dimension
    CLASS = "class"  # classification which is the argmax of 'pred'
    LOGITS = "logits"  # logits of classifications before applying softmax or sigmoid
    SEGMENTATIONS = "segmentation"
    CLASSNAMES = "classnames"
    COORD = "coord"


class DataProcessorParamsMeta(ABCMeta):
    def __subclasscheck__(cls, subclass):
        # Custom subclass check for correct 'issubclass' support
        # I do not yet know why this is required...
        return cls == subclass or any(cls == sc for sc in subclass.__mro__)


def is_valid_sample(sample: Sample, mode: PipelineMode) -> bool:
    if sample is None:
        return False
    if sample.inputs is None and mode in INPUT_PROCESSOR:
        return False
    if sample.targets is None and mode in TARGETS_PROCESSOR:
        return False
    return True


@pai_dataclass
@dataclass
class DataProcessorParams(ABC, metaclass=DataProcessorParamsMeta):
    """
    Parameters for a DataProcessor
    Implement to add additional parameters and to define the actual DataProcessor class (`cls()`)

    If the input pipeline is run in parallel, only the DataProcessorParams are copied to the spawned sub-processes
    which are then used to `create()` the actual data Processor in the thread.
    Therefore, the __init__ function of a DataProcessor is only called in its actual thread.
    """

    modes: Set[PipelineMode] = field(
        default_factory=GENERAL_PROCESSOR.copy,
        metadata=pai_meta(
            help="The PipelineModes when to apply this DataProcessor (e.g., only during PipelineMode.TRAINING)"
        ),
    )

    @staticmethod
    def cls() -> Type["DataProcessorBase"]:
        raise NotImplementedError

    def create(self, data_params: "DataBaseParams", mode: PipelineMode, **kwargs) -> "DataProcessorBase":
        """
        Create the actual DataProcessor
        """
        if mode not in self.modes:
            raise ValueError(
                f"{self.__class__.__name__} should not be created since the pipeline mode {mode} is not in its "
                f"modes {[m.value for m in self.modes]}"
            )
        try:
            return self.cls()(params=self, data_params=data_params, mode=mode, **kwargs)
        except TypeError as e:
            logger.exception(e)
            raise TypeError(
                f"Data processor of type {self.cls()} could not be instantiated. Maybe the init function has a wrong "
                f"signature."
            ) from e


T = TypeVar("T", bound=DataProcessorParams)


class DataProcessorBase(Generic[T], ABC):
    """
    Base class for a data processor which is used to transform Samples to Samples possibly in a separate process.
    A DataProcessorBase must be instantiated by its parameters.

    Do not implement DataProcessorBase but instead either MappingDataProcessor or GeneratingDataProcessor.

    See Also:
        MappingDataProcessor
        GeneratingDataProcessor
    """

    def __init__(self, params: T, data_params: "DataBaseParams", mode: PipelineMode, **kwargs):
        super().__init__()
        assert len(kwargs) == 0, kwargs
        assert isinstance(params, DataProcessorParams)
        self.params: T = params
        self.data_params = data_params
        self.mode = mode

    def supports_preload(self):
        return True

    def is_valid_sample(self, sample: Sample) -> bool:
        return is_valid_sample(sample, self.mode)

    @typechecked
    def __call__(self, sample: Union[Sample, Iterable[Sample]]) -> Union[Sample, Iterable[Sample]]:
        if isinstance(self, MappingDataProcessor):
            return self.apply(sample)
        elif isinstance(self, GeneratingDataProcessor):
            return self.generate(sample)
        else:
            raise TypeError("A DataProcessor must inherit either MappingDataProcessor or GeneratingDataProcessor")

    def preload(
        self,
        samples: List[Sample],
        num_processes=1,
        drop_invalid=True,
        progress_bar=False,
    ) -> Iterable[Sample]:
        return self.apply_on_samples(samples, num_processes, drop_invalid, progress_bar)

    @abstractmethod
    def apply_on_samples(
        self,
        samples: Iterable[Sample],
        num_processes=1,
        drop_invalid=True,
        progress_bar=False,
    ) -> Iterator[Sample]:
        raise NotImplementedError


class GeneratingDataProcessor(DataProcessorBase[T]):
    """
    A GeneratingDataProcessor realizes a n-to-m mapping of Samples, that is, this data processor can consume and
    produce an arbitrary number of samples.

    Note, that running this processor in parallel is only supported during training since the outputs is are
    not ordered.
    """

    @abstractmethod
    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        raise NotImplementedError

    def apply_on_samples(
        self,
        samples: Iterable[Sample],
        num_processes=1,
        drop_invalid=True,
        progress_bar=False,
    ) -> Iterator[Sample]:
        mapped = self.generate(samples)
        if drop_invalid:
            mapped = filter(self.is_valid_sample, mapped)
        return mapped


class MappingDataProcessor(DataProcessorBase[T]):
    """
    The MappingDataProcessor is the most common case for a DataProcessor because it implements a simple 1-to-1 mapping
    of Samples.
    Hereby, a Simple transformation is applied on a single Sample.
    """

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        raise NotImplementedError

    def apply_on_samples(
        self,
        samples: Iterable[Sample],
        num_processes=1,
        drop_invalid=True,
        progress_bar=False,
    ) -> Iterator[Sample]:
        mapped = parallel_map(
            self.apply_on_sample,
            samples,
            processes=num_processes,
            progress_bar=progress_bar,
            desc=f"Applying data processor {self.__class__.__name__}",
        )
        if drop_invalid:
            mapped = filter(self.is_valid_sample, mapped)
        return mapped

    def apply_on_sample(self, sample: Sample) -> Sample:
        if sample.meta is None:
            sample = sample.new_meta({})
        return self.apply(sample.copy())


class SequenceProcessorParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["DataProcessorBase"]:
        raise NotImplementedError


class SequenceProcessor(MappingDataProcessor[SequenceProcessorParams]):
    """
    The SequenceProcessor (internal usage in tfaip only!) groups multiple MappingDataProcessors into one DataProcessor
    which is then passed to the actual Pipeline.
    """

    def apply(self, sample: Sample) -> Optional[Sample]:
        # Apply the complete list of data processors
        # Non valid samples return None

        if sample.meta is None:
            sample = sample.new_meta({})

        if not self.is_valid_sample(sample):
            return None

        for p in self.processors:
            sample = p(sample)
            if not self.is_valid_sample(sample):
                return None

        return sample

    def __init__(self, params: "DataBaseParams", mode: PipelineMode, processors: List[MappingDataProcessor]):
        super().__init__(params=SequenceProcessorParams(), data_params=params, mode=mode)
        self.processors = processors
