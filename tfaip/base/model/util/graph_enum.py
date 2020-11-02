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
