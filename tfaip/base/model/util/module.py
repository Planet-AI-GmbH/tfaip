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
import importlib
import inspect
import pkgutil
from typing import List, Type
from tfaip.base.model import GraphBase


def import_submodules(package, recursive=True):
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    if not hasattr(package, '__path__'):
        # is already a module
        results[package.__name__] = package
        return results

    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def import_graphs(module_name, sub_module_name='graphs') -> List[Type[GraphBase]]:
    modules = import_submodules(module_name[:module_name.rfind('.')] + '.' + sub_module_name)
    return [c for n, c in sum([inspect.getmembers(module, lambda member: inspect.isclass(
        member) and member.__module__ == module.__name__ and issubclass(member, GraphBase)) for _, module in
                               modules.items()], [])]


