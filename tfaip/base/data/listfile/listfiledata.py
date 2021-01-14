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
from random import Random
from typing import Iterable, Type

from tfaip.base.data.data import DataBase
from tfaip.base.data.databaseparams import DataGeneratorParams
from tfaip.base.data.listfile.listfiledataparams import ListFilePipelineParams, ListFileDataParams
from tfaip.base.data.pipeline.datapipeline import DataGenerator, DataPipeline
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample
from tfaip.base.data.listfile.data_list_helpers import ListMixDefinition, FileListProviderFn, FileListIterablor
from tfaip.util.math.iter_helpers import ThreadSafeIterablor


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

    def generate(self) -> Iterable[Sample]:
        iterator = self._create_iterator()
        if self.params.limit > 0:
            iterator = itertools.islice(iterator, self.params.limit)

        return map(lambda fn: Sample(inputs=fn, targets=fn), iterator)


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
