from abc import ABC, abstractmethod


class DataWorker(ABC):
    @abstractmethod
    def initialize_thread(self):
        pass

    @abstractmethod
    def process(self, *args, **kwargs):
        pass
