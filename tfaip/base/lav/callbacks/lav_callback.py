from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tfaip.base.lav.lav import LAV
    from tfaip.base.model.modelbase import ModelBase
    from tfaip.base.data.data import DataBase


class LAVCallback(ABC):
    def __init__(self):
        self.lav: 'LAV' = None  # Set from lav
        self.data: 'DataBase' = None  # Set from lav
        self.model: 'ModelBase' = None  # set from lav

    @abstractmethod
    def on_sample_end(self, inputs, targets, outputs):
        pass

    @abstractmethod
    def on_step_end(self, inputs, targets, outputs, metrics):
        pass

    @abstractmethod
    def on_lav_end(self, result):
        pass
