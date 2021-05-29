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
"""Implementation of the LAVParams"""
from dataclasses import dataclass, field
from typing import Union, List

from paiargparse import pai_meta, pai_dataclass

from tfaip import PipelineMode
from tfaip.data.databaseparams import DataPipelineParams
from tfaip import DeviceConfigParams


@pai_dataclass
@dataclass
class LAVParams:
    """
    Parameters for LAV
    """

    pipeline: DataPipelineParams = field(
        default_factory=DataPipelineParams, metadata=pai_meta(fix_dc=True, mode="ssnake")
    )
    model_path: Union[str, List[str]] = field(default=None, metadata=pai_meta(mode="ignore"))
    device: DeviceConfigParams = field(default_factory=DeviceConfigParams)
    silent: bool = field(default=False, metadata=pai_meta(help="Suppress model prediction print to console/log"))
    store_results: bool = field(default=True, metadata=pai_meta(help="Save lav results (metrics) in "))

    def __post_init__(self):
        self.pipeline.mode = PipelineMode.EVALUATION
