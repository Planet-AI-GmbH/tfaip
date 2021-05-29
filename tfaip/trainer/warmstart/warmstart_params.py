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
"""Definition of the WarmStartParams"""
from dataclasses import dataclass, field
from typing import List, Optional

from paiargparse import pai_meta, pai_dataclass


@pai_dataclass
@dataclass
class WarmStartParams:
    """Parameters for warm-starting from a model."""

    model: Optional[str] = field(
        default=None, metadata=pai_meta(help="Path to the saved model or checkpoint to load the weights from.")
    )

    allow_partial: bool = field(default=False, metadata=pai_meta(help="Allow that not all weights can be matched."))
    trim_graph_name: bool = field(
        default=True,
        metadata=pai_meta(
            help="Remove the graph name from the loaded model and the target model. This is useful if the model name "
            "changed"
        ),
    )

    rename: List[str] = field(
        default_factory=list,
        metadata=pai_meta(
            help="A list of renaming rules to perform on the loaded weights. Format: FROM->TO FROM->TO ..."
        ),
    )

    add_suffix: str = field(default="", metadata=pai_meta(help="Add suffix str to all variable names"))

    rename_targets: List[str] = field(
        default_factory=list,
        metadata=pai_meta(
            help="A list of renaming rules to perform on the target weights. Format: FROM->TO FROM->TO ..."
        ),
    )

    exclude: Optional[str] = field(
        default=None, metadata=pai_meta(help="A regex applied on the loaded weights to ignore from loading.")
    )
    include: Optional[str] = field(
        default=None, metadata=pai_meta(help="A regex applied on the loaded weights to include from loading.")
    )

    auto_remove_numbers_for: List[str] = field(default_factory=lambda: ["lstm_cell"])
