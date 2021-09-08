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
from typing import Iterable
import logging

import tensorflow as tf

from tfaip import Sample

logger = logging.getLogger(__name__)


def unbatched(values, batch_size=None):
    if batch_size is None:
        flatted_values = tf.nest.flatten(values)
        if len(flatted_values) == 0:
            return []

        batch_size = flatted_values[0].shape[0]

    def extract_at(v, i):
        if v.shape[0] != batch_size:
            logger.warning(
                f"Expected batch size {batch_size} but got {v.shape[0]} (total shape {v.shape})."
                f"This tensor will be added as is. "
                f"If it is an input or output, consider to compute this input in the graph. "
                f"Else do not return it as output, or call broad_cast to the actual batch size. "
            )
            return v
        return v[i]

    return [tf.nest.map_structure(lambda x: extract_at(x, i), values) for i in range(batch_size)]


def to_unbatched_samples(inputs, targets, outputs, meta) -> Iterable[Sample]:
    flatted_values = tf.nest.flatten(meta) + tf.nest.flatten(outputs)
    batch_size = flatted_values[0].shape[0] if len(flatted_values) > 0 else None

    if inputs is not None:
        inputs = unbatched(inputs, batch_size)

    if targets is not None:
        targets = unbatched(targets, batch_size)

    if outputs is not None:
        outputs = unbatched(outputs, batch_size)

    if meta is not None:
        meta = unbatched(meta, batch_size)

    batch_size = len(inputs or targets or outputs or meta)

    for i in range(batch_size):
        yield Sample(
            inputs=inputs[i] if inputs else None,
            targets=targets[i] if targets else None,
            outputs=outputs[i] if outputs else None,
            meta=meta[i] if meta else None,
        )
