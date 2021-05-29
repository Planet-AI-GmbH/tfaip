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
"""Definition of the EarlyStoppingParams"""
from dataclasses import dataclass, field
from typing import Optional

from paiargparse import pai_dataclass, pai_meta


@pai_dataclass
@dataclass
class EarlyStoppingParams:
    """EarlyStoppingParameters"""

    # User parameters
    best_model_output_dir: Optional[str] = field(
        default=None, metadata=pai_meta(help="Override the default output_dir of the best model.")
    )
    best_model_name: str = field(default="best", metadata=pai_meta(help="Name of the best model."))
    frequency: int = field(
        default=1,
        metadata=pai_meta(
            help="Frequency in terms of epochs when to test for a new best model. Defaults to 1, i.e. after each epoch."
        ),
    )
    n_to_go: int = field(
        default=-1,
        metadata=pai_meta(
            help="Set to a value > 0 to enable early stopping, i.e. if not better model was found after n_to_go epochs "
            "(modify by frequency), training is stopped."
        ),
    )
    lower_threshold: float = field(
        default=-1e100,
        metadata=pai_meta(
            help="Threshold that must be reached at least (if mode=max) to count for early stopping, "
            "or stop training immediately (if mode=min) if the monitored value is lower. E.g. 0 for an accuracy."
        ),
    )
    upper_threshold: float = field(
        default=1e100,
        metadata=pai_meta(
            help="If mode=min the monitored value must be lower to count for early stopping, "
            "or if mode=max and the threshold is exceeded training is stopped immediately. E.g. 1 for an accuracy."
        ),
    )

    # Internal parameters
    # Do not set them manually since they will be overwritten
    # * mode and monitor are set by ModelBase
    # * current and n track the current state of early stopping to allow for restoring
    mode: Optional[str] = field(
        default=None, metadata=pai_meta(mode="ignore", help="Either max or min. Set by the model")
    )
    current: Optional[float] = field(
        default=None, metadata=pai_meta(mode="ignore", help="The current monitored value.")
    )  # If loaded from a checkpoint, this value is already set
    monitor: Optional[str] = field(
        default=None, metadata=pai_meta(mode="ignore", help="The resolved log value to monitor. Set by the model.")
    )
    n: int = field(
        default=1,
        metadata=pai_meta(
            mode="ignore",
            help="The current number of epochs (setup by by frequency) without an improvement. "
            "If n reached n_to_go, training will be stopped.",
        ),
    )
