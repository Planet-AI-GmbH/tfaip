import enum

from tfaip.util.enum import StrEnum


class Padding(StrEnum):
    Valid = "VALID"  # valid padding
    Same = "SAME"  # same padding
