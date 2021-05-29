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
from typing import Dict, Optional, Iterable

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

    def register_field_value(self, field, value, recursive=True):
        resource_id = field.metadata.get("resource_id", field.name) if field.metadata else field.name
        if isinstance(value, Resource):
            self.register(resource_id, value)
        elif field.type == Resource or field.type == Optional[Resource]:
            if isinstance(value, str):
                self.register(resource_id, Resource(value))
            elif value is None:
                return  # Unset resource
            else:
                raise ValueError(f"Value of a Resource must be of type str but got {value} of type {type(value)}.")
        elif is_dataclass(value):
            if recursive:
                self.register_all(value)

    def register_all(self, params, recursive=True):
        for field in fields(params):
            value = getattr(params, field.name)
            if isinstance(value, Iterable):
                value = list(value)
                for v in value:
                    self.register_field_value(field, v, recursive=recursive)
            else:
                self.register_field_value(field, value, recursive=recursive)

    def register(self, r_id: str, resource: Resource):
        if r_id in self._resources:
            raise KeyError(f"A resource with id {r_id} already exists.")

        if not resource.initialized:
            if resource.abs_path is None:
                resource.abs_path = os.path.abspath(os.path.join(self.working_dir, resource.rel_path))

            resource.basename = os.path.basename(resource.rel_path)
            resource.dump_path = os.path.join(self.dump_prefix_dir, resource.dump_dir, resource.basename)

        resource.initialized = True
        self._resources[r_id] = resource
        return resource

    def remove(self, res: str):
        del self._resources[res]

    def items(self):
        return self._resources.items()

    def __contains__(self, item: str):
        return item in self._resources

    def get(self, resource_id: str) -> Resource:
        return self._resources[resource_id]

    def dump(self, location: str):
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
