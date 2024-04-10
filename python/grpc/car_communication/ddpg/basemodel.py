from torch.functional import Tensor
import torch.nn as nn
import torch

from typing import Any, Callable


class Base(nn.Module):
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
        self.activation = nn.ReLU()

    def forward_linear_block(
        self,
        input_tensor:Tensor, 
        prefix:str, 
        count:int,
    ) -> Tensor:
        '''count = 3 -> 1,2,3'''
        x = input_tensor
        for index in range(1, count+1):
            linear = self._get_linear_by_name(prefix=prefix, index=index)
            bn = self._get_batchnorm_by_name(prefix=prefix, index=index)
            # processing forward
            if bn:
                x = bn(linear(x))
            else:
                x = linear(x)
            x = self.activation(x)
        return x
    
    def forward(self, *args, **kwargs) -> Tensor: #pyright: ignore
        raise NotImplementedError   
    
    @_eval_mode
    def predict(self, *args, **kwargs) -> Tensor:
        '''enable eval mode and get prediction'''
        tensor = self.forward(*args, **kwargs)
        return tensor

    def _get_layer_by_name(self, name:str) -> Any | None:
        try:
            layer = self.__getattr__(name)
            return layer
        except:
            return None

    def _get_linear_by_name(self, prefix: str, index:int) -> nn.Linear:
        name = f'{prefix}_fc{index}'
        layer = self._get_layer_by_name(name=name)
        if not(layer): raise Exception(f'Failed get linear block: {name}')
        return layer

    def _get_batchnorm_by_name(self, prefix:str, index:int) -> nn.BatchNorm1d | None:
        name = f'{prefix}_bn{index}'
        layer = self._get_layer_by_name(name=name)
        return layer

class BaseModel(Base):
    _image_flatten_size: int = 3 * 3 * 256

    def __init__(self, action_dim: int, max_action: int = 1, *args, **kwargs) -> None:
        super().__init__(action_dim, max_action, *args, **kwargs)

        self._init_image_nn()
        self._init_distance_nn()
        self._init_routers_nn()
        self._init_speed_nn()
        self._init_stage1_nn()
        self._init_stage2_nn()

    def image_to_input(self, image: Tensor) -> Tensor:
        # conv layer 1
        x = self.image_conv1(image)
        x = self.image_bn1(x)
        x = self.activation(x)
        x = self.image_pool1(x)

        # conv layer 2
        x = self.image_conv2(x)
        x = self.image_bn2(x)
        x = self.activation(x)
        x = self.image_pool2(x)

        # conv layer 3
        x = self.image_conv3(x)
        x = self.image_bn3(x)
        x = self.activation(x)
        x = self.image_pool3(x)

        # conv layer 4
        x = self.image_conv4(x)
        x = self.image_bn4(x)
        x = self.activation(x)
        x = self.image_pool4(x)

        # flatten
        x = x.view(-1, self._image_flatten_size)

        # fc layer 1
        x = self.image_fc1(x)
        x = self.activation(x)

        # fc layer 2
        x = self.image_fc2(x)
        x = self.activation(x)

        # fc layer 3
        x = self.image_fc3(x)
        x = self.activation(x)
        return x

    def distance_to_input(
        self, 
        distance_sensors_distances: Tensor,
        *,
        prefix:str='distance',
        count:int=1,
    ) -> Tensor:
        x = self.forward_linear_block(
            input_tensor=distance_sensors_distances,
            prefix=prefix,
            count=count,
        )
        return x

    def routers_to_input(
        self, 
        nearest_routers: Tensor,
        *,
        prefix:str='routers',
        count:int=1,
    ) -> Tensor:
        x = self.forward_linear_block(
            input_tensor=nearest_routers,
            prefix=prefix,
            count=count,
        )
        return x

    def speed_to_input(
        self, 
        speed: Tensor,
        *,
        prefix:str='speed',
        count:int=1,
    ) -> Tensor:
        x = self.forward_linear_block(
            input_tensor=speed,
            prefix=prefix,
            count=count,
        )
        return x

    def stage1_to_input(
        self,
        in_target_area: Tensor,
        distance_to_target_router: Tensor,
        *,
        prefix:str = 'stage1',
        count:int = 1,
    ) -> Tensor:
        _c = [in_target_area, distance_to_target_router]
        concat = torch.cat(_c, dim=1)
        x = self.forward_linear_block(
            input_tensor=concat,
            prefix=prefix,
            count=count,
        )
        return x

    def stage2_to_input(
        self,
        in_target_area: Tensor,
        boxes_is_found: Tensor,
        target_found: Tensor,
        distance_to_box: Tensor,
        *,
        prefix:str='stage2',
        count:int=1,
    ) -> Tensor:
        _c = [
            in_target_area,
            boxes_is_found,
            target_found,
            distance_to_box,
        ]
        concat = torch.cat(_c, dim=1)
        x = self.forward_linear_block(
            input_tensor=concat,
            prefix=prefix,
            count=count,
        )
        return x


    def _init_image_nn(self, output_dim: int = 8) -> None:
        self.image_conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=2, padding=2)
        self.image_bn1 = nn.BatchNorm2d(32)
        self.image_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.image_conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=2, padding=2)
        self.image_bn2 = nn.BatchNorm2d(64)
        self.image_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.image_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.image_bn3 = nn.BatchNorm2d(128)
        self.image_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.image_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.image_bn4 = nn.BatchNorm2d(256)
        self.image_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.image_fc1 = nn.Linear(self._image_flatten_size, 512)
        self.image_fc2 = nn.Linear(512, 128)
        self.image_fc3 = nn.Linear(128, output_dim)

    def _init_distance_nn(self, input_dim:int = 6, output_dim: int = 8) -> None:
        self.distance_fc1 = nn.Linear(input_dim, output_dim)

    def _init_routers_nn(self, input_dim:int = 3, output_dim:int = 8) -> None:
        self.routers_fc1 = nn.Linear(input_dim, output_dim)

    def _init_speed_nn(self, input_dim:int = 1, output_dim:int = 8) -> None:
        self.speed_fc1 = nn.Linear(input_dim, output_dim)

    def _init_stage1_nn(self, input_dim:int = 2, output_dim:int = 8) -> None:
        self.stage1_fc1 = nn.Linear(input_dim, output_dim)

    def _init_stage2_nn(self, input_dim:int = 4, output_dim:int = 8) -> None:
        self.stage2_fc1 = nn.Linear(input_dim, output_dim)
