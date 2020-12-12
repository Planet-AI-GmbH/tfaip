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
from abc import ABC, abstractmethod
from typing import List


class JoinableHolder:
    def __init__(self):
        self._joinables: List[Joinable] = []

    def register_joinable(self, joinable: 'Joinable'):
        self._joinables.append(joinable)

    def withdraw_joinable(self, joinable):
        if joinable in self._joinables:
            self._joinables.remove(joinable)

    def join(self):
        while len(self._joinables) > 0:
            self._joinables[0].join()

        self._joinables = []


class Joinable(ABC):
    def __init__(self, holder: JoinableHolder):
        self._holder = holder
        holder.register_joinable(self)

    @abstractmethod
    def join(self):
        self._holder.withdraw_joinable(self)
