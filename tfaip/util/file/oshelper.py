import os


class ChDir:
    def __init__(self, path):
        self._cd = os.getcwd()
        self.path = path

    def __enter__(self):
        self._cd = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._cd)
