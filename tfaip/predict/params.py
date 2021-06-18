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
"""Definition of the PredictorParams"""
from dataclasses import dataclass, field

from paiargparse import pai_meta, pai_dataclass
from tfaip.data.databaseparams import DataPipelineParams
from tfaip import DeviceConfigParams


@pai_dataclass
@dataclass
class PredictorParams:
    """
    Parameters for the PredictorBase.
    """

    device: DeviceConfigParams = field(default_factory=DeviceConfigParams, metadata=pai_meta(fix_dc=True))
    pipeline: DataPipelineParams = field(
        default_factory=DataPipelineParams, metadata=pai_meta(fix_dc=True, mode="ssnake")
    )

    silent: bool = field(
        default=False,
        metadata=pai_meta(help="If silent, do not print the current predicted sample. See also progress_bar."),
    )
    progress_bar: bool = field(default=True, metadata=pai_meta(help="Render a progress bar during prediction."))
    run_eagerly: bool = field(
        default=False, metadata=pai_meta(help="Run the prediction model in eager mode. Use for debug only.")
    )
    include_targets: bool = field(
        default=False,
        metadata=pai_meta(
            mode="ignore", help="Include the targets which must be present (!) of each sample to the predicted sample."
        ),
    )
    include_meta: bool = field(default=False, metadata=pai_meta(mode="ignore"))
