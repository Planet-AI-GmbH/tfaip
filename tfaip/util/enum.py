import enum


class StrEnum(str, enum.Enum):
    """ Enum class with strings that is json serializable """
