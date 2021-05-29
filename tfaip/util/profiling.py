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
"""Definitions for profiling simple functions

Either wrap a function:
```
@profile
def my_func():
    ...
```

or inline code:
```
with ProfileScope("my_scope_name"):
    ...
```

At some point at the code call `print_profiling`
"""
import os
import time
from functools import wraps

import numpy as np

PROF_DATA = {}
ENABLE = int(os.environ.get("TFAIP_PROFILING", "0"))
FORMAT = os.environ.get("TFAIP_PROFILING_FORMAT", "<9.3")


def enable_profiling(enable: bool):
    global ENABLE
    ENABLE = 1 if enable else 0


def format_profiling(format_str: str):
    global FORMAT
    FORMAT = format_str


class MeasureTime:
    """Class to measure the execution time in a with block

    Usage:
        with MeasureTime() as t:
            ...
        print(t.duration)
    """

    def __init__(self):
        self.start = 0
        self.end = 0
        self.duration = -1

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start

    def duration_till_now(self):
        if self.duration != -1:
            return self.duration  # already measured
        return time.time() - self.start


def profile(fn):
    if not ENABLE:
        return fn

    @wraps(fn)
    def with_profiling(*args, **kwargs):
        with ProfileScope(fn.__module__ + "::" + fn.__name__):
            ret = fn(*args, **kwargs)

        return ret

    return with_profiling


class ProfileScope(MeasureTime):
    """Scope to measure the time and automatically write it to the profiling storage

    Usage:
        with ProfileScope("my_scope_label"):
            ...
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not ENABLE:
            return
        super().__exit__(exc_type, exc_val, exc_tb)
        if self.name not in PROF_DATA:
            PROF_DATA[self.name] = []
        l = PROF_DATA[self.name]
        l.append(self.duration)
        if len(l) > 1000:
            del l[0]


def print_profiling(print_fn=print):
    if not ENABLE:
        return
    print_fn(f"found {len(PROF_DATA)} functions")
    # leng = max([len(s) for s in PROF_DATA.keys()])
    mytuple = [(fname, points, np.sum(points)) for fname, points in PROF_DATA.items()]
    mytuple = sorted(mytuple, key=lambda t: -t[2])
    for fname, points, s in mytuple:
        max_time = max(points)
        avg_time = s / len(points)
        m = np.median(points)
        print_fn(
            f"cnt = {len(points):4} | tot = {s:{FORMAT}} | max = {max_time:{FORMAT}} "
            f"| avg = {avg_time:{FORMAT}} | med = {m:{FORMAT}} | name = {fname}"
        )
