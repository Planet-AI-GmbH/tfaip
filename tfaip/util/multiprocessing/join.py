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
"""Definition of a Joinable and a JoinableHolder"""
from abc import ABC, abstractmethod
from typing import List


class JoinableHolder:
    """The join-able holder stores a list of Joinables

    When join() is called, the JoinableHolder waits for all joinables to join() until resuming.
    This is useful for waiting for processes to finish.
    """

    def __init__(self):
        self._joinables: List[Joinable] = []

    def register_joinable(self, joinable: "Joinable"):
        self._joinables.append(joinable)

    def withdraw_joinable(self, joinable):
        if joinable in self._joinables:
            self._joinables.remove(joinable)

    def join(self):
        while len(self._joinables) > 0:
            joinable = self._joinables[0]
            joinable.join()
            self.withdraw_joinable(joinable)  # this should be called on joinable.join()

        self._joinables = []


class Joinable(ABC):
    def __init__(self, holder: JoinableHolder):
        self._holder = holder
        holder.register_joinable(self)

    @abstractmethod
    def join(self):
        self._holder.withdraw_joinable(self)
