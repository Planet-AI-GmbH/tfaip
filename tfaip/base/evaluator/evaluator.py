from abc import ABC, abstractmethod

from tfaip.base.data.pipeline.definitions import InputOutputSample


class Evaluator(ABC):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def handle(self, sample: InputOutputSample):
        raise NotImplementedError
