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
"""Definition of a Resource used in the ResourceManager"""
from dataclasses import field
from typing import Optional

from dataclasses_json import config


def resource_field(*, resource_id: Optional[str] = None, **kwargs):
    kwargs["metadata"] = {**kwargs.get("metadata", {}), **config(encoder=Resource.encode, decoder=Resource.decode)}
    if resource_id:
        kwargs["resource_id"] = resource_id
    return field(**kwargs)


class Resource:
    """
    Resource of a scenario.

    Declare a resource in a dataclass by adding
    ```
    charmaplists= = field(default=None,
                              metadata={**pai_meta(help="File specifying the character map used", required=True),
                                        **config(encoder=Resource.encode, decoder=Resource.decode)}
    ```

    A resource is instantiated with an (abs) path to the file and must always be added to a resource manager.
    Call abs_path to obtain the path of the resource for usage.
    """

    @staticmethod
    def encode(r: "Resource"):
        if r is None:
            return None
        return r.initial_path

    @staticmethod
    def decode(r: str):
        if r is None:
            return None
        return Resource(r)

    def __init__(self, initial_path: str):
        self.initial_path = initial_path

        self.rel_path: str = initial_path

        self.initialized: bool = False
        self.dump_dir: str = ""

        # These values will be specified from the resource manager
        self.abs_dump_path: Optional[str] = None
        self.basename: Optional[str] = None
        self.dump_path: Optional[str] = None
        self.abs_path: Optional[str] = None

    def __str__(self) -> str:
        assert self.initialized
        return self.abs_path or "Not initialized"
