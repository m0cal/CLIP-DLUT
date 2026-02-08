from enum import StrEnum, auto

class TaskStatus(StrEnum):
    PENDING = auto()
    PROCESSING = auto()
    FINISHED = auto()
    STOPPED = auto()
    FAILED = auto()

class LUTFormat(StrEnum):
    CUBE = auto()
    PNG = auto()
