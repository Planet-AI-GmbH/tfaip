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
"""Utilities to handle randomness"""
import logging

logger = logging.getLogger(__name__)


def set_global_random_seed(n):
    """
    Set all random seeds:
    - PYTHONHASHSEED
    - numpy
    - tensorflow
    - random
    """
    if n is None:
        return

    import os  # pylint: disable=import-outside-toplevel
    from numpy.random import seed  # pylint: disable=import-outside-toplevel
    import random  # pylint: disable=import-outside-toplevel
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    logger.info(f"Setting all random seed to {n}")
    os.environ["PYTHONHASHSEED"] = str(n)
    seed(n + 1)
    tf.random.set_seed(n + 2)
    random.seed(n + 3)
