from torch.functional import Tensor
import torch.nn as nn
import torch

from .basemodel import BaseModel


class ActorModel(BaseModel):
    def __init__(self, action_dim:int, max_action:int = 1) -> None:
        super().__init__(action_dim=action_dim, max_action=max_action)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.fc1_10 = nn.Linear(1, 10)
        self.fc2_10 = nn.Linear(2, 10)
        self.fc3_10 = nn.Linear(3, 10)
        self.fc6_10 = nn.Linear(6, 10)
        
        self.fc10_128 = nn.Linear(10,128)
        self.fc40_400 = nn.Linear(5 * 10, 400)
        self.fc400_128 = nn.Linear(400, 128)

        self.fc_concat = nn.Linear(3*256, 256)
        self.fc256_32 = nn.Linear(256, 32)
        self.fc_out = nn.Linear(32, action_dim)

        self.bn256 = nn.BatchNorm1d(256)
        self.bn400 = nn.BatchNorm1d(400)

    def forward(
        self, 
        # car parameters
        speed: Tensor,
        steer: Tensor,
        forward: Tensor,
        # car sensors  [0, 1]
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        distance_to_target_router: Tensor,
        nearest_routers: Tensor,
        distance_to_box: Tensor, 
        # car hints [0 or 1]
        in_target_area: Tensor, 
        boxes_is_found: Tensor, 
        target_found: Tensor,
        *args, **kwargs #pyright: ignore
    ) -> Tensor:
        x_speed = self.relu(self.fc1_10(speed))
        x_routers = self.relu(self.fc3_10(nearest_routers))
        x_distance = self.relu(self.fc6_10(distance_sensors_distances))
        _c = torch.cat([in_target_area, distance_to_target_router], dim=1)
        x_stage1 = self.relu(self.fc2_10(_c))
        _c = torch.cat([in_target_area, distance_to_box, boxes_is_found], dim=1)
        x_stage2 = self.relu(self.fc3_10(_c))

        # concat 5 * 10
        concat = torch.cat([x_speed, x_distance, x_routers, x_stage1, x_stage2], dim=1) 
        concat = self.relu(self.bn400(self.fc40_400(concat))) #400
        concat = self.relu(self.fc400_128(concat)) #128

        x_steer = self.fc10_128(self.fc1_10(steer))
        x_forward = self.fc10_128(self.fc1_10(forward))
        x_target = self.fc10_128(self.fc1_10(target_found))

        x1 = torch.cat([concat, x_steer], dim=1) # 256 
        x2 = torch.cat([concat, x_forward], dim=1) # 256 
        x3 = torch.cat([concat, x_target], dim=1) # 256

        x = self.fc_concat(torch.cat([x1,x2,x3], dim=1)) # 256
        x = self.relu(self.bn256(x))

        x = self.relu(self.fc256_32(x))

        outputs = self.tanh(self.fc_out(x))
        return self.max_action * outputs
