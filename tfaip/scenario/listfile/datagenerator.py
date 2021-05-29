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
"""Definition of the ListFileDataGenerator"""
import logging
from random import Random
from typing import Iterable

from tfaip import PipelineMode, Sample
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.scenario.listfile.data_list_helpers import ListMixDefinition, FileListProviderFn
from tfaip.scenario.listfile.params import ListsFileGeneratorParams

logger = logging.getLogger(__name__)


class ListsFileDataGenerator(DataGenerator[ListsFileGeneratorParams]):
    """
    DataGenerator of ListFiles, a text file where each line is a sample.

    On training, random samples (file names) will be chosen respecting the list rations (if multiple lists are passes),
    see ListMixDefinition.

    On any other mode, the samples will be yielded in Order.
    """

    def __len__(self):
        return len(self._create_iterator())

    def _create_iterator(self):
        if self.mode == PipelineMode.TRAINING:
            # shuffle data and pick of list ratios
            file_names_it = ListMixDefinition(
                list_filenames=self.params.lists, mixing_ratio=self.params.list_ratios
            ).as_generator(Random())
        else:
            # Simply load all file names in order
            if self.params.list_ratios and any(r != 1 for r in self.params.list_ratios):
                logger.warning(
                    "List ratio was specified by is only used during training. "
                    "Do not set list ratios to remove this warning."
                )
            file_names_it = sum(
                [FileListProviderFn(train_list_fn).get_list() for train_list_fn in self.params.lists], []
            )

        return file_names_it

    def generate(self) -> Iterable[Sample]:
        iterator = self._create_iterator()
        return map(lambda fn: Sample(inputs=fn, targets=fn), iterator)
