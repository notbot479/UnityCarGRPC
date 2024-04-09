from torch.functional import Tensor
import torch.nn as nn

from typing import Callable


class BaseModel(nn.Module):
    @staticmethod
    def _eval_mode(func) -> Callable:
        def wrapper(self, *args,**kwargs) -> None:
            self.eval()
            result = func(self, *args, **kwargs)
            self.train()
            return result
        return wrapper

    def __init__(self, action_dim:int, max_action:int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.max_action = max_action
    
    def forward(self, *args, **kwargs) -> Tensor: #pyright: ignore
        raise NotImplementedError   
    
    @_eval_mode
    def predict(self, *args, **kwargs) -> Tensor:
        '''enable eval mode and get prediction'''
        tensor = self.forward(*args, **kwargs)
        return tensor
