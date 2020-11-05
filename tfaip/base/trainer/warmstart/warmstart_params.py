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
from typing import List

from dataclasses_json import dataclass_json

from tfaip.util.argument_parser import dc_meta


@dataclass_json
@dataclass
class WarmstartParams:
    model: str = field(default=None, metadata=dc_meta(
        help="Path to the saved model or checkpoint to load the weights from."
    ))

    allow_partial: bool = field(default=False, metadata=dc_meta(
        help="Allow that not all weights can be matched."
    ))
    trim_graph_name: bool = field(default=True, metadata=dc_meta(
        help="Remove the graph name from the loaded model and the target model. This is useful if the model name "
             "changed"
    ))
    rename: List[str] = field(default_factory=list, metadata=dc_meta(
        help="A list of renaming rules to perform on the weights. Format: [FROM->TO,FROM->TO,...]"
    ))

    exclude: str = field(default=None, metadata=dc_meta(
        help="A regex applied on the loaded weights to ignore from loading."
    ))
    include: str = field(default=None, metadata=dc_meta(
        help="A regex applied on the loaded weights to include from loading."
    ))

