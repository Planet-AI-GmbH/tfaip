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
"""Definition of the TFAIPArgumentParser

The TFAIPArgumentParser is a PAIArgumentParser but allows to drop "Params" suffixes when selecting classes.
"""
from dataclasses import is_dataclass
from typing import Type, List

from paiargparse import PAIArgumentParser
from paiargparse.dataclass_parser import PAIDataClassArgumentParser


class TFAIPDataClassArgumentParser(PAIDataClassArgumentParser):
    """
    Extension to the PAI argument parser, to allow aliases without `Params` suffix.
    """

    def alt_names_of_choice(self, choice) -> List[str]:
        names = super().alt_names_of_choice(choice)
        # allow to specify without Params suffix
        if choice.__name__.endswith("Params"):
            names.append(choice.__name__[:-6])
        return names


class TFAIPArgumentParser(PAIArgumentParser):
    def _data_class_argument_parser_cls(self) -> Type[PAIDataClassArgumentParser]:
        return TFAIPDataClassArgumentParser


def post_init(dc):
    """
    Recursively apply __post_init__ on a given dataclass
    """
    if not is_dataclass(dc):
        return

    # recursively call post init on a dataclass
    for name in dc.__dataclass_fields__.keys():
        value = getattr(dc, name)
        post_init(value)

    if hasattr(dc, "__post_init__"):
        dc.__post_init__()
