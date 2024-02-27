from dataclasses import dataclass
from typing import Union
from enum import Enum

Meter = Rssi = float
Score = Union[float, int]
Pixel = int

class Done(Enum):
    _ = 1
    HIT_OBJECT = 2
    TARGET_IS_FOUND = 3

    def __bool__(self) -> bool:
        return self != Done._

@dataclass
class Status:
    ok: bool
    msg: str
