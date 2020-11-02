from collections import Iterator, Iterable
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

    def next(self):
        if not self._repeat and self._index + 1 == len(self._list):
            raise StopIteration
        self._index = (self._index + 1) % len(self._list)
        return self._list[self._index]

    def __next__(self):
        return self.next()


class ThreadSafeIterablor(Iterator, Iterable):
    """ wraps iterators to add thread safeness """

    def __init__(self, it: Iterator):
        assert isinstance(it, Iterator)
        self._lock = Lock()
        self._iter = it.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return self._iter.__next__()
