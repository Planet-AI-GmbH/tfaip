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
from dataclasses import dataclass


class Resource:
    @staticmethod
    def encode(r: 'Resource'):
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
        self.dump_dir: str = ''

        # These values will be specified from the resource manager
        self.basename: str = None
        self.dump_path: str = None
        self.abs_path: str = None

    def __str__(self):
        assert self.initialized
        return self.abs_path

