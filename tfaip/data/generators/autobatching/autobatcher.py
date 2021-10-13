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
from typing import Tuple, Dict, List, Union
import numpy as np


def auto_batch_to_same_samples_per_batch(
    batch_bins: int,
    utt2shapes: List[Dict[str, List[Union[float, int]]]],
    min_batch_size: int = 1,
    sort_in_batch: str = "descending",
    sort_batch: str = "ascending",
    drop_last: bool = False,
    padding: bool = True,
) -> List[Tuple[str, ...]]:
    """Auto batch data to the same samples per batch

    Params:
        utt2shapes: The outer list is the number of data sources (e.g. input data and ground truth),
            The dictionary is a mapping from sample IDs to shapes whereby the shape

    Returns:
        A list of batches identified by the ids
    """
    first_utt2shape = utt2shapes[0]

    # Sort samples in ascending order
    # (shape order should be like (Length, Dim))
    keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
    if len(keys) == 0:
        raise RuntimeError("0 lines found")
    if padding:
        # If padding case, the feat-dim must be same over whole corpus,
        # therefore the first sample is referred
        feat_dims = [np.prod(d[keys[0]][1:]) for d in utt2shapes]
    else:
        feat_dims = None

    # Decide batch-sizes
    batch_sizes = []
    current_batch_keys = []
    for key in keys:
        current_batch_keys.append(key)
        # shape: (Length, dim1, dim2, ...)
        if padding:
            for d in utt2shapes:
                if tuple(d[key][1:]) != tuple(d[keys[0]][1:]):
                    raise RuntimeError("If padding=True, the feature dimension must be unified.")
            bins = sum(len(current_batch_keys) * sh[key][0] * d for sh, d in zip(utt2shapes, feat_dims))
        else:
            bins = sum(np.prod(d[k]) for k in current_batch_keys for d in utt2shapes)

        if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
            batch_sizes.append(len(current_batch_keys))
            current_batch_keys = []
    else:
        if len(current_batch_keys) != 0 and (not drop_last or len(batch_sizes) == 0):
            batch_sizes.append(len(current_batch_keys))

    if len(batch_sizes) == 0:
        # Maybe we can't reach here
        raise RuntimeError("0 batches")

    # If the last batch-size is smaller than minimum batch_size,
    # the samples are redistributed to the other mini-batches
    if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
        for i in range(batch_sizes.pop(-1)):
            batch_sizes[-(i % len(batch_sizes)) - 1] += 1

    if not drop_last:
        # Bug check
        assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

    # Set mini-batch
    batch_list = []
    iter_bs = iter(batch_sizes)
    bs = next(iter_bs)
    minibatch_keys = []
    for key in keys:
        minibatch_keys.append(key)
        if len(minibatch_keys) == bs:
            if sort_in_batch == "descending":
                minibatch_keys.reverse()
            elif sort_in_batch == "ascending":
                # Key are already sorted in ascending
                pass
            else:
                raise ValueError("sort_in_batch must be ascending" f" or descending: {sort_in_batch}")

            batch_list.append(tuple(minibatch_keys))
            minibatch_keys = []
            try:
                bs = next(iter_bs)
            except StopIteration:
                break

    if sort_batch == "ascending":
        pass
    elif sort_batch == "descending":
        batch_list.reverse()
    else:
        raise ValueError(f"sort_batch must be ascending or descending: {sort_batch}")

    return batch_list
