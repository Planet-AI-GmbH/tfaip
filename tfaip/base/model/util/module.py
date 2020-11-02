import importlib
import inspect
import pkgutil
from typing import List, Type
from tfaip.base.model import GraphBase


def import_submodules(package, recursive=True):
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def import_graphs(module_name, sub_module_name = 'graphs') -> List[Type[GraphBase]]:
    modules = import_submodules(module_name[:module_name.rfind('.')] + '.' + sub_module_name)
    return [c for n, c in sum([inspect.getmembers(module, lambda member: inspect.isclass(
        member) and member.__module__ == module.__name__ and issubclass(member, GraphBase)) for _, module in
                               modules.items()], [])]


