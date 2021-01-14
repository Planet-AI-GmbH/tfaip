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
from dataclasses import dataclass, field
from typing import Union, List

from dataclasses_json import dataclass_json

from tfaip.base.device_config import DeviceConfigParams
from tfaip.util.argumentparser import dc_meta


@dataclass_json
@dataclass
class LAVParams:
    model_path: Union[str, List[str]] = field(default=None, metadata=dc_meta(arg_mode='ignore'))
    device_params: DeviceConfigParams = field(default_factory=lambda: DeviceConfigParams())
    silent: bool = False
