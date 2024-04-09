from typing import Union, Literal
from torch import Tensor


def shift_left(tensor: Tensor, max_action:int = 1) -> Tensor:
    tensor = (2 * tensor) - max_action
    return tensor

def shift_right(tensor: Tensor, max_action:int = 1) -> Tensor:
    tensor = (tensor + max_action) / 2
    return tensor

def minmaxscale(
    value:float, 
    min_val:float, 
    max_val:float,
    *,
    round_factor: int = 7
) -> float:
    if min_val == max_val: return 0
    scaled_value = (value - min_val) / (max_val - min_val)
    return round(scaled_value, round_factor)

def zeroOrOne(data:bool) -> Union[Literal[0], Literal[1]]:
    if not(isinstance(data, bool)): return 0
    return 1 if data else 0
