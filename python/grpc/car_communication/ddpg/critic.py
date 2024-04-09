from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch

from .basemodel import BaseModel


class CriticModel(BaseModel):
    def __init__(self, action_dim:int, max_action:int = 1) -> None:
        super().__init__(action_dim=action_dim, max_action=max_action)

        self.relu = nn.ReLU()

        self.fc1_10 = nn.Linear(1, 10)
        self.fc2_10 = nn.Linear(2, 10)
        self.fc3_10 = nn.Linear(3, 10)
        self.fc6_10 = nn.Linear(6, 10)
        
        self.fc10_128 = nn.Linear(10,128)
        self.fc50_500 = nn.Linear((5+1) * 10, 500)
        self.fc500_128 = nn.Linear(500, 128)
        self.fc128_32 = nn.Linear(128, 32)

        self.fc_concat = nn.Linear(3*128, 128)
        self.fc256_32 = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)  # outputs

        self.bn128 = nn.BatchNorm1d(128)
        self.bn256 = nn.BatchNorm1d(256)
        self.bn500 = nn.BatchNorm1d(500)


    def forward(
        self, 
        # car parameters
        speed: Tensor, 
        # car sensors [0, 1]
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        nearest_routers: Tensor,
        distance_to_target_router: Tensor,
        distance_to_box: Tensor, 
        # car hints [0 or 1]
        in_target_area: Tensor, 
        boxes_is_found: Tensor, 
        #critic input [-1, 1]
        actor_action: Tensor, 
        *args,**kwargs #pyright: ignore
    ) -> Tensor:
        x_speed = self.relu(self.fc1_10(speed))
        x_distance = self.relu(self.fc6_10(distance_sensors_distances))
        x_routers = self.relu(self.fc3_10(nearest_routers))
        _c = torch.cat([in_target_area, distance_to_target_router], dim=1)
        x_stage1 = self.relu(self.fc2_10(_c))
        _c = torch.cat([in_target_area, distance_to_box, boxes_is_found], dim=1)
        x_stage2 = self.relu(self.fc3_10(_c))
        x_actor = self.relu(self.fc2_10(actor_action))

        # 6 * 10
        concat = torch.cat([x_speed, x_distance, x_routers, x_stage1, x_stage2, x_actor], dim=1) 
        concat = self.relu(self.bn500(self.fc50_500(concat))) #500
        concat = self.relu(self.fc500_128(concat)) #128
        concat = self.relu(self.fc128_32(concat))

        outputs = self.fc_out(concat)
        return outputs
