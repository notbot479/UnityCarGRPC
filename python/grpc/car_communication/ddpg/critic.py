from torch.functional import Tensor
import torch.nn as nn
import torch

from .basemodel import BaseModel


class CriticModel(BaseModel):
    _concat_tensor = 5 * 8

    def __init__(
        self, 
        action_dim:int, 
        max_action:int = 1, 
        mock_image:bool = False,
    ) -> None:
        params = {
            'action_dim': action_dim,
            'max_action': max_action,
            'mock_image': mock_image,
        }
        super().__init__(**params)

        self.concat_fc = nn.Linear(self._concat_tensor, 1024)

        self._init_actor_action_nn()
        self._init_concat_nn()

    @staticmethod
    def normalize_actor_action(actor_action: Tensor) -> Tensor:
        a = actor_action.clone()
        a[a < 0] = 0
        a = abs(a)
        b = actor_action.clone()
        b[b > 0] = 0
        b = abs(b)
        concat = torch.cat([a,b], dim=1)
        return concat

    def forward(
        self, 
        # car parameters [0, 1]
        speed: Tensor, 
        # car sensors [0, 1]
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        distance_to_target_router: Tensor,
        # car hints [0 or 1]
        in_target_area: Tensor, 
        #critic input [-1, 1]
        actor_action: Tensor, 
        *args,**kwargs #pyright: ignore
    ) -> Tensor:
        # normalize inputs
        x_image = self.image_to_input(image=image)
        x_distance = self.distance_to_input(
            distance_sensors_distances=distance_sensors_distances,
        )
        x_speed = self.speed_to_input(speed=speed)
        x_stage1 = self.stage1_to_input(
            in_target_area=in_target_area,
            distance_to_target_router=distance_to_target_router,
        )
        actor_action = self.normalize_actor_action(actor_action=actor_action)
        x_actor = self.actor_action_to_input(actor_action=actor_action)
        # merge inputs to concat
        concat = self.inputs_to_concat(
            x_image=x_image,
            x_distance=x_distance,
            x_speed=x_speed,
            x_stage1=x_stage1,
            x_actor=x_actor,
        )
        concat = self.concat_fc(concat)
        concat = self.activation(concat)
        q = self.concat_to_q(concat=concat)
        return q

    def actor_action_to_input(
        self, 
        actor_action: Tensor,
        *,
        prefix:str='actor_action',
        count:int=1,
    ) -> Tensor:
        x = self.forward_linear_block(
            input_tensor=actor_action,
            prefix=prefix,
            count=count,
        )
        return x

    def concat_to_q(
        self, 
        concat: Tensor,
        *,
        prefix:str='concat',
        count:int=5,
    ) -> Tensor:
        x = self.forward_linear_block(
            input_tensor=concat,
            prefix=prefix,
            count=count,
        )
        # convert to q value (no activation func)
        q = self.concat_fc_out(x)
        return q

    def inputs_to_concat(
        self,
        x_image: Tensor,
        x_distance: Tensor,
        x_speed: Tensor,
        x_stage1: Tensor,
        x_actor: Tensor,
    ) -> Tensor:
        _c = [x_image, x_distance, x_speed, x_stage1, x_actor]
        x = torch.cat(_c, dim=1)
        return x 


    def _init_concat_nn(self, input_dim:int = 1024, output_dim:int = 256) -> None:
        self.concat_fc1 = nn.Linear(input_dim, 2048)
        self.concat_bn1 = nn.BatchNorm1d(2048)
        
        self.concat_fc2 = nn.Linear(2048, 2048)
        self.concat_bn2 = nn.BatchNorm1d(2048)
        
        self.concat_fc3 = nn.Linear(2048, 1024)
        self.concat_bn3 = nn.BatchNorm1d(1024)
        
        self.concat_fc4 = nn.Linear(1024, 512)
        self.concat_bn4 = nn.BatchNorm1d(512)

        self.concat_fc5 = nn.Linear(512, output_dim)
        self.concat_fc_out = nn.Linear(output_dim, 1)

    def _init_actor_action_nn(self, input_dim:int = 4, output_dim:int = 8) -> None:
        self.actor_action_fc1 = nn.Linear(input_dim, output_dim)
