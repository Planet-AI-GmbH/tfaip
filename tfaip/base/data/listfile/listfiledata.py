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
import itertools
from abc import ABC
from dataclasses import dataclass, field, fields
from random import Random
from typing import List, Iterable, Type

from dataclasses_json import dataclass_json

from tfaip.base.data.data import DataBase
from tfaip.base.data.data_base_params import DataBaseParams, DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import DataGenerator, DataPipeline
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, InputTargetSample
from tfaip.util.argument_parser import dc_meta
from tfaip.base.data.listfile.data_list_helpers import ListMixDefinition, FileListProviderFn, FileListIterablor
from tfaip.util.math.iter_helpers import ThreadSafeIterablor


@dataclass_json
@dataclass
class ListsFilePipelineParams(DataGeneratorParams):
    lists: List[str] = field(default=None, metadata=dc_meta(
        help="Training list files."
    ))
    list_ratios: List[float] = field(default=None, metadata=dc_meta(
        help="Ratios of picking list files. Must be supported by the scenario"
    ))

    def validate(self):
        super(ListsFilePipelineParams, self).validate()

        if self.lists:
            if not self.list_ratios:
                self.list_ratios = [1.0] * len(self.lists)
            else:
                if len(self.list_ratios) != len(self.lists):
                    raise ValueError(f"Length of list_ratios must be equals to number of lists. Got {self.list_ratios}!={self.lists}")

    def split(self) -> List['ListFilePipelineParams']:
        out = []
        for l in self.lists:
            t = ListFilePipelineParams()
            for f in fields(self.__class__):
                if f.name in {'lists', 'list_ratios'}:
                    continue
                setattr(t, f.name, getattr(self, f.name))
            t.list = l
            out.append(t)
        return out


@dataclass_json
@dataclass
class ListFilePipelineParams(DataGeneratorParams):
    list: str = field(default=None, metadata=dc_meta(
        help="The validation list for testing the model during training."
    ))


@dataclass_json
@dataclass
class ListFileDataParams(DataBaseParams):
    train: ListsFilePipelineParams = field(default_factory=ListsFilePipelineParams, metadata=dc_meta(arg_mode='snake'))
    val: ListFilePipelineParams = field(default_factory=ListFilePipelineParams, metadata=dc_meta(arg_mode='snake'))
    lav: ListsFilePipelineParams = field(default_factory=ListsFilePipelineParams, metadata=dc_meta(arg_mode='snake'))


class ListsFileDataGenerator(DataGenerator):
    def __len__(self):
        if self.params.limit > 0:
            return self.params.limit
        return len(self._create_iterator())

    def _create_iterator(self):
        if self.mode == PipelineMode.Training:
            # One instance of train dataset to produce infinite many samples
            # Setting up the TRAIN
            file_names_it = ListMixDefinition(list_filenames=self.params.lists,
                                              mixing_ratio=self.params.list_ratios).get_as_generator(Random())
        else:
            a = FileListProviderFn(self.params.list)
            iterablor = FileListIterablor(a, False)
            file_names_it = ThreadSafeIterablor(iterablor)

        return file_names_it

    def generate(self) -> Iterable[InputTargetSample]:
        iterator = self._create_iterator()
        if self.params.limit > 0:
            iterator = itertools.islice(iterator, self.params.limit)

        return map(lambda fn: InputTargetSample(fn, fn), iterator)


class ListsFileDataPipeline(DataPipeline, ABC):
    def create_data_generator(self) -> DataGenerator:
        return ListsFileDataGenerator(self.mode, self.generator_params)


class ListFileData(DataBase, ABC):
    @classmethod
    def data_processor_factory(cls) -> DataProcessorFactory:
        return DataProcessorFactory([])

    @classmethod
    def prediction_generator_params_cls(cls) -> Type[DataGeneratorParams]:
        return ListFilePipelineParams

    @classmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        class ImplementedClass(ListsFileDataPipeline):
            @staticmethod
            def data_cls() -> Type['DataBase']:
                return cls

        return ImplementedClass

    def _list_lav_dataset(self) -> Iterable:
        params: ListFileDataParams = self.params()
        if not params.lav.lists:
            if params.val.list:
                return super(ListFileData, self)._list_lav_dataset()
            else:
                raise ValueError("No LAV and no VAL list were given. Cannot load lav dataset.")
        return (self.create_pipeline(PipelineMode.Evaluation, p) for p in params.lav.split())
