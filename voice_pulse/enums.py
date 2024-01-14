from enum import Enum, auto


class VadAggressiveness(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    HIGHEST = auto()


class VadEngine(Enum):
    WEBRTC = auto()
    SILERIO = auto()
