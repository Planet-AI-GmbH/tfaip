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
import logging
import os
from dataclasses import dataclass, field
from numbers import Number
from random import Random
from typing import List, Optional

from tfaip.util.file.io import DefaultIOContext, IOContext
from tfaip.util.math.blend_helpers import BlendIterablor
from tfaip.util.math.iter_helpers import ListIterablor, ThreadSafeIterablor

logger = logging.getLogger(__name__)


@dataclass
class ListMixDefinition:
    list_filenames: List[str]
    mixing_ratio: List[float]
    io_context: IOContext = field(default_factory=lambda: DefaultIOContext())

    def __post_init__(self):
        assert len(self.list_filenames) == len(self.mixing_ratio)
        assert len(self.list_filenames) > 0
        assert isinstance(self.list_filenames, list)
        assert isinstance(self.mixing_ratio, list)
        for list_filename in self.list_filenames:
            assert isinstance(list_filename, str)
            if not os.path.isfile(self.io_context.get_io_filepath(list_filename)):
                raise FileNotFoundError("File '{}' does not exist!".format(self.io_context.get_io_filepath(list_filename)))
        for part in self.mixing_ratio:
            assert isinstance(part, Number)

    def get_as_generator(self, rnd: Random):
        train_flis = [FileListIterablor(FileListProviderFn(train_list_fn)) for train_list_fn in self.list_filenames]
        return ThreadSafeIterablor(BlendIterablor(train_flis, self.mixing_ratio, rnd))


class FileListProviderFn(object):
    def __init__(self, file_name: str):
        self._file_name = file_name
        self._logger = logger.getChild(str(type(self)))
        self._file_list = None

    def _load_list(self):
        retval = []
        with open(self._file_name, 'r') as list_file:
            for img_fn in [line.rstrip('\n ') for line in list_file]:
                if img_fn is not None and len(img_fn) > 0:
                    retval.append(img_fn)
        return retval

    def get_list(self) -> Optional[List[str]]:
        """
        :param io_context:
        :type io_context: IOContext
        :param cache:
        :type cache: dict
        :return:
        :rtype: list of tuples (filename, reference/transcript, length/width)
        """
        if self._file_name is None:
            return None
        if self._file_list is None:
            self._file_list = self._load_list()
        else:
            self._logger.info("reusing: " + self._file_name)
        return self._file_list


class FileListIterablor(ListIterablor):
    def __init__(self, file_list_provider: FileListProviderFn, repeat=True):
        super(FileListIterablor, self).__init__(file_list_provider.get_list(), repeat=repeat)
