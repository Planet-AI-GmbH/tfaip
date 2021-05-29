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
from collections.abc import Iterator, Iterable
from random import Random
from threading import Lock
from typing import List


class ListIterablor(Iterator, Iterable):
    def __init__(self, l: List, repeat: bool = True):
        self._list = l
        self._index = -1
        self._repeat = repeat

    def shuffle(self, rng: Random):
        rng.shuffle(self._list)

    def __iter__(self):
        return self

    def __len__(self):
        if self._list is None:
            return 0
        return len(self._list)

    def next(self):
        if not self._repeat and self._index + 1 == len(self._list):
            raise StopIteration
        self._index = (self._index + 1) % len(self._list)
        return self._list[self._index]

    def __next__(self):
        return self.next()


class ThreadSafeIterablor(Iterator, Iterable):
    """wraps iterators to add thread safeness"""

    def __init__(self, it: Iterator):
        assert isinstance(it, Iterator)
        self._lock = Lock()
        self._it = it
        self._iter = it.__iter__()

    def __len__(self):
        return len(self._it)

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return self._iter.__next__()
