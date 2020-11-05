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

logger = logging.getLogger(__name__)


def set_global_random_seed(n):
    if n is None:
        return

    logger.info(f"Setting all random seed to {n}")
    import os
    os.environ['PYTHONHASHSEED'] = str(n)
    from numpy.random import seed
    seed(n + 1)
    import tensorflow as tf
    tf.random.set_seed(n + 2)
    import random
    random.seed(n + 3)
