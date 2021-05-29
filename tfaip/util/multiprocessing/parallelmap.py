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
"""Definition of a parallel map with an optional progress bar"""
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
from typing import Iterable, Any

from tqdm import tqdm


def tqdm_wrapper(iterable, *, total=1, desc="", progress_bar=False):
    if not progress_bar:
        return iterable
    else:
        return tqdm(iterable, total=total, desc=desc)


def parallel_map(
    f, d, *, desc="", processes=1, progress_bar=False, use_thread_pool=False, max_tasks_per_child=None
) -> Iterable[Any]:
    if processes <= 0:
        processes = os.cpu_count()

    if len(d) == 0:
        # Nothing to to
        return []

    processes = min(processes, len(d))

    if processes == 1:
        if progress_bar:
            out = list(tqdm(map(f, d), desc=desc, total=len(d)))
        else:
            out = list(map(f, d))

    else:
        if use_thread_pool:
            with ThreadPool(processes=processes) as pool:
                if progress_bar:
                    out = list(tqdm(pool.imap(f, d), desc=desc, total=len(d)))
                else:
                    out = pool.map(f, d)
        else:
            with multiprocessing.Pool(processes=processes, maxtasksperchild=max_tasks_per_child) as pool:
                if progress_bar:
                    out = list(tqdm(pool.imap(f, d), desc=desc, total=len(d)))
                else:
                    out = pool.map(f, d)

    return out
