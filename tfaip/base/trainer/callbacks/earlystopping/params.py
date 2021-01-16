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
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EarlyStoppingParams:
    # Logging parameters -> set from ModelBase
    mode_: str = None
    current_: float = None  # If loaded from a checkpoint, this value is already set
    monitor_: str = None
    n_: int = -1

    # User parameters
    # TODO: help
    best_model_output_dir: Optional[str] = None
    best_model_name: str = 'best'
    frequency: int = 1
    n_to_go: int = -1

    # Thresholds: either ignore in early stopping, or stop immediately
    # e.g. 0 and 1 for accuracy
    lower_threshold: float = -1e100
    upper_threshold: float = 1e100