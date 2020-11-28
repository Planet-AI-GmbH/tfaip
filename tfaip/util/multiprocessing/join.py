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
