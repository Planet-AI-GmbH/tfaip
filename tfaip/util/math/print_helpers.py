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
from typing import Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def printable_of_batched_dict(v: Dict[str, Any]) -> Dict[str, Any]:
    """Basically compute the mean of each element

    If there is no mean, the element is skipped with 'nan'
    """

    out = {}
    for k, v in v.items():
        try:
            out[k] = np.mean(v)
        except TypeError:
            out[k] = "nan"
        except Exception as e:
            logging.exception(e)
            logging.warning(f"An unknown exception occurred when converting {k}={v} to a string")
            out[k] = "Unknown exception"

    return out
