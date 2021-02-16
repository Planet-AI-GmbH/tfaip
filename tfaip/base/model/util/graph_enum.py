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
from typing import Type, List, Union

from tfaip.base.model import GraphBase
from tfaip.base.model.util.module import import_graphs
from tfaip.util.enum import StrEnum


def create_graph_enum(graphs: Union[str, List[Type[GraphBase]]]):
    if isinstance(graphs, str):
        graphs = import_graphs(graphs)
    names = {g.__name__: g.__name__ for g in graphs}

    class GraphEnum(StrEnum):
        @property
        def cls(self):
            return [g for g in graphs if g.__name__ == self.value][0]

    return GraphEnum('Graphs', names)
