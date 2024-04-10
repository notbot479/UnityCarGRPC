from torch.functional import Tensor
import torch.nn as nn
import torch

from .basemodel import BaseModel


class ActorModel(BaseModel):
    def __init__(self, action_dim:int, max_action:int = 1) -> None:
        super().__init__(action_dim=action_dim, max_action=max_action)
        self.output_func = nn.Tanh()

        self.concat_fc1 = nn.Linear(6 * 8, 64)

        self._init_forward_nn()
        self._init_steer_nn()

    def forward(
        self, 
        # car parameters
        speed: Tensor,
        # car sensors
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        distance_to_target_router: Tensor,
        nearest_routers: Tensor,
        distance_to_box: Tensor, 
        # car hints
        in_target_area: Tensor, 
        boxes_is_found: Tensor, 
        target_found: Tensor, 
        *args, **kwargs #pyright: ignore
    ) -> Tensor:
        # normalize inputs
        x_image = self.image_to_input(image=image)
        x_distance = self.distance_to_input(
            distance_sensors_distances=distance_sensors_distances,
        )
        x_routers = self.routers_to_input(nearest_routers=nearest_routers)
        x_speed = self.speed_to_input(speed=speed)
        x_stage1 = self.stage1_to_input(
            in_target_area=in_target_area,
            distance_to_target_router=distance_to_target_router,
        )
        x_stage2 = self.stage2_to_input(
            in_target_area=in_target_area,
            boxes_is_found=boxes_is_found,
            target_found=target_found,
            distance_to_box=distance_to_box,
        )

        # merge inputs to concat
        concat = self.inputs_to_concat(
            x_image=x_image,
            x_distance=x_distance,
            x_routers=x_routers,
            x_speed=x_speed,
            x_stage1=x_stage1,
            x_stage2=x_stage2,
        )

        # concat dense
        concat = self.concat_fc1(concat)
        concat = self.activation(concat)

        x1 = self.forward_to_action(concat=concat)
        x2 = self.steer_to_action(concat=concat)
        
        outputs = torch.cat([x1,x2], dim=1)
        return self.max_action * outputs

    def inputs_to_concat(
        self,
        x_image: Tensor,
        x_distance: Tensor,
        x_routers: Tensor,
        x_speed: Tensor,
        x_stage1: Tensor,
        x_stage2: Tensor,
    ) -> Tensor:
        _c = [x_image, x_distance, x_routers, x_speed, x_stage1, x_stage2]
        x = torch.cat(_c, dim=1)
        return x

    def steer_to_action(
        self, 
        concat: Tensor,
        *,
        prefix:str='steer',
        count:int=3,
    ) -> Tensor:
        x = self.forward_linear_block(
            input_tensor=concat,
            prefix=prefix,
            count=count,
        )
        x = self.steer_fc_out(x)
        action = self.output_func(x)
        return action

    def forward_to_action(
        self, 
        concat: Tensor,
        *,
        prefix:str='forward',
        count:int=3,
    ) -> Tensor:
        x = self.forward_linear_block(
            input_tensor=concat,
            prefix=prefix,
            count=count,
        )
        x = self.forward_fc_out(x)
        action = self.output_func(x)
        return action


    def _init_steer_nn(self, input_dim:int = 64, output_dim:int = 64) -> None:
        self.steer_fc1 = nn.Linear(input_dim, 256)
        self.steer_fc2 = nn.Linear(256, 128)
        self.steer_fc3 = nn.Linear(128, output_dim)

        self.steer_fc_out = nn.Linear(output_dim, 1)

    def _init_forward_nn(self, input_dim:int = 64, output_dim:int = 64) -> None:
        self.forward_fc1 = nn.Linear(input_dim, 256)
        self.forward_fc2 = nn.Linear(256, 128)
        self.forward_fc3 = nn.Linear(128, output_dim)

        self.forward_fc_out = nn.Linear(output_dim, 1)
