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
""" Definition of a shared memory queue for (numeric) numpy arrays

The queue will store numpy arrays in shared memory instead of pickling them (which is 5 times slower).
"""
import multiprocessing
from typing import NamedTuple, List, Any

import numpy as np

from tfaip import Sample

SHARED_MEMORY_SUPPORT = hasattr(multiprocessing, "shared_memory")


def create_queue_with_fallback(context, maxsize, fallback=True):
    """creates a shared memory queue but with a normal multiprocessing queue as fallback if not supported."""
    if SHARED_MEMORY_SUPPORT:
        return SharedMemoryQueue(maxsize=maxsize, context=context)
    else:
        if fallback:
            return context.Queue(maxsize=maxsize)
        else:
            raise NotImplementedError


class SharedMemoryNumpyArray(NamedTuple):
    name: str
    shape: List[int]
    dtype: Any


def numpy_to_shared_memory(data: np.ndarray) -> SharedMemoryNumpyArray:
    shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_memory.buf[:] = data.tobytes()
    shared_memory.close()
    return SharedMemoryNumpyArray(shared_memory.name, data.shape, data.dtype)


def numpy_from_shared_memory(data: SharedMemoryNumpyArray, unlink=True) -> np.ndarray:
    mem = multiprocessing.shared_memory.SharedMemory(create=False, name=data.name)
    data = np.frombuffer(mem.buf, dtype=data.dtype).reshape(data.shape).copy()
    mem.close()
    if unlink:
        mem.unlink()
    return data


def to_shared_memory(data_struct):
    if isinstance(data_struct, Sample):
        data_struct = (
            data_struct.new_inputs(to_shared_memory(data_struct.inputs))
            .new_targets(to_shared_memory(data_struct.targets))
            .new_outputs(to_shared_memory(data_struct.outputs))
        )
    elif isinstance(data_struct, dict):
        data_struct = {k: to_shared_memory(v) for k, v in data_struct.items()}
    elif isinstance(data_struct, list):
        data_struct = list(map(to_shared_memory, data_struct))
    elif isinstance(data_struct, tuple):
        data_struct = tuple(map(to_shared_memory, data_struct))
    elif isinstance(data_struct, np.ndarray):
        if np.issubdtype(data_struct.dtype, np.number):
            data_struct = numpy_to_shared_memory(data_struct)

    return data_struct


def from_shared_memory(data_struct):
    if isinstance(data_struct, Sample):
        data_struct = (
            data_struct.new_inputs(from_shared_memory(data_struct.inputs))
            .new_targets(from_shared_memory(data_struct.targets))
            .new_outputs(from_shared_memory(data_struct.outputs))
        )
    elif isinstance(data_struct, dict):
        data_struct = {k: from_shared_memory(v) for k, v in data_struct.items()}
    elif isinstance(data_struct, list):
        data_struct = list(map(from_shared_memory, data_struct))
    elif isinstance(data_struct, SharedMemoryNumpyArray):
        data_struct = numpy_from_shared_memory(data_struct)
    elif isinstance(data_struct, tuple):
        data_struct = tuple(map(from_shared_memory, data_struct))

    return data_struct


class SharedMemoryQueue:
    """A Queue that used shared memory instead of pickle to store numpy arrays."""

    def __init__(self, maxsize=0, context=multiprocessing):
        self.maxsize = maxsize
        self.queue = context.Queue(maxsize)

    def put(self, data, block=True, timeout=None):
        self.queue.put(to_shared_memory(data), block, timeout)

    def get(self, block=True, timeout=None):
        data = self.queue.get(block=block, timeout=timeout)
        return from_shared_memory(data)

    def empty(self):
        return self.queue.empty()

    def close(self):
        return self.queue.close()
