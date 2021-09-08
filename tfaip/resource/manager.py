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
"""Definition of the ResourceManager"""
import logging
import os
import shutil
from dataclasses import fields, is_dataclass
from typing import Dict, Optional, Iterable, Tuple, Union

from tfaip.resource.resource import Resource

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    A ResourceManager (as used in Data) will handle `Resources`, i.e. files, that are required to run an exported model.
    For example, the tokenizer or character map.
    This will help to ensure that resources are found after exporting.

    The `DataBase`-class already provides an implemented ResourceManager.

    Usage:
        - create a ResourceManager and set the current working dir (first argument)
        - register resources (either register_all, or register)
        - dump the resources to a new location.
    """

    def __init__(self, working_dir: str, dump_prefix_dir: str = "resources"):
        self.working_dir = working_dir if working_dir is not None else os.getcwd()
        self.dump_prefix_dir = dump_prefix_dir
        self._resources: Dict[str, Resource] = {}
        self._resource_prefixes: Dict[
            str, Tuple[Union[str, int]]
        ] = {}  # relative location of a resource within the params

    def register_field_value(self, field, value, prefixes: Tuple[Union[str, int]], recursive=True):
        resource_id = field.metadata.get("resource_id", field.name) if field.metadata else field.name
        assert "/" not in resource_id, f"'/' is a non-allowed character in {resource_id} of field {field.name}."
        if isinstance(value, Resource):
            self.register(resource_id, value, prefixes)
        elif field.type == Resource or field.type == Optional[Resource]:
            if isinstance(value, str):
                self.register(resource_id, Resource(value), prefixes)
            elif value is None:
                return  # Unset resource
            else:
                raise ValueError(f"Value of a Resource must be of type str but got {value} of type {type(value)}.")
        elif is_dataclass(value):
            if recursive:
                self.register_all(value, prefixes=prefixes)

    def register_all(self, params, recursive=True, prefixes=tuple()):
        for field in fields(params):
            value = getattr(params, field.name)
            if isinstance(value, Iterable):
                value = list(value)
                for i, v in enumerate(value):
                    self.register_field_value(
                        field,
                        v,
                        recursive=recursive,
                        prefixes=prefixes
                        + (
                            field.name,
                            i,
                        ),
                    )
            else:
                self.register_field_value(field, value, recursive=recursive, prefixes=prefixes + (field.name,))

    def register(self, r_id: str, resource: Resource, prefixes: Tuple[Union[str, int]]):
        if r_id in self._resources:
            raise KeyError(f"A resource with id {r_id} already exists.")

        resource.__initialize__(self.working_dir, self.dump_prefix_dir)

        resource.initialized = True
        self._resources[r_id] = resource
        self._resource_prefixes[r_id] = prefixes
        return resource

    def remove(self, res: str):
        del self._resources[res]

    def items(self):
        return self._resources.items()

    def __contains__(self, item: str):
        return item in self._resources

    def get(self, resource_id: str) -> Resource:
        return self._resources[resource_id]

    def dump(self, location: str, dump_dict: Optional[dict] = None):
        for r_id, resource in self._resources.items():
            dump_dir = os.path.join(location, self.dump_prefix_dir, resource.dump_dir)
            abs_dump_path = os.path.join(location, resource.dump_path)
            resource.abs_dump_path = abs_dump_path  # store last abs dump path
            logger.debug(f"Exporting resource {r_id} to {dump_dir}")
            os.makedirs(dump_dir, exist_ok=True)
            if os.path.isdir(resource.abs_path):
                if not os.path.exists(abs_dump_path):
                    shutil.copytree(resource.abs_path, abs_dump_path)
            else:
                shutil.copy(resource.abs_path, abs_dump_path)

        if dump_dict is not None:
            self.convert_to_dump_dict(dump_dict)

    def convert_to_dump_dict(self, struct: dict):
        """Update all resource path in the dict to dump paths.

        This is required for exporting to make "relative" paths.
        """
        for r_id, resource in self.items():
            self.assign_to_dict_struct(r_id, struct, resource.dump_path)

    def assign_to_dict_struct(self, r_id, struct: dict, value):
        """Utility function to assign a value to a dict structure"""
        steps = self._resource_prefixes[r_id]
        for i in range(len(steps) - 1):
            if isinstance(struct, (dict, list)):
                struct = struct[steps[i]]
            else:
                raise TypeError(f"{struct} must be a dict or list to modify {r_id} to value {value}")

        struct[steps[-1]] = value
