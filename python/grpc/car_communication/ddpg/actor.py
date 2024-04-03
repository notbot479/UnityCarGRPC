from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch

from .normalization import shift_range


class ActorModel(nn.Module):
    def __init__(
        self, 
        action_dim:int, 
        max_action:int = 1, 
        *,
        shift_range:bool=False,
    ) -> None:
        super().__init__()
        self.max_action = max_action
        self.shift_range = shift_range
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=16, stride=2, padding=7)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=8, stride=2, padding=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(3*3*512, 256)

        self.fc64_256 = nn.Linear(64, 256)  
        self.fc256_64 = nn.Linear(256, 64)  
        self.fc2 = nn.Linear(1, 64)  # speed, steer, forward, target
        self.fc3 = nn.Linear(6, 64)  # distance_sensors_distances
        self.fc4 = nn.Linear(2, 64)  # stage 1
        self.fc5 = nn.Linear(3, 64)  # stage 2

        self.fc6 = nn.Linear(5*256, 256)  # concatenated
        self.fc7 = nn.Linear(512, 256)  # x1,x2,x3
        self.fc8 = nn.Linear(3*64, 32)  # concatenated2

        self.fc9 = nn.Linear(32, action_dim)  # outputs

    def forward(
        self, 
        # car parameters
        speed: Tensor, 
        steer: Tensor, 
        forward: Tensor, 
        # car sensors
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        distance_to_target_router: Tensor,
        distance_to_box: Tensor, 
        # car hints
        in_target_area: Tensor, 
        boxes_is_found: Tensor, 
        target_found: Tensor,
        *args, **kwargs #pyright: ignore
    ) -> Tensor:
        if self.shift_range:
            steer = shift_range(steer, max_action=self.max_action)
            forward = shift_range(forward, max_action=self.max_action)

        x_img = F.relu(self.bn1(self.conv1(image)))
        x_img = F.max_pool2d(x_img, 2)
        x_img = F.relu(self.bn2(self.conv2(x_img)))
        x_img = F.max_pool2d(x_img, 2)
        x_img = F.relu(self.bn3(self.conv3(x_img)))
        x_img = F.max_pool2d(x_img, 2)
        x_img = F.relu(self.bn4(self.conv4(x_img)))
        x_img = F.max_pool2d(x_img, 2)
        x_img = x_img.view(x_img.size(0), -1)  # Flatten
        x_img = F.relu(self.fc1(x_img))

        x_speed = F.relu(self.fc2(speed))
        x_speed = F.relu(self.fc64_256(x_speed))

        x_distance = F.relu(self.fc3(distance_sensors_distances))
        x_distance = F.relu(self.fc64_256(x_distance))

        _c = torch.cat([distance_to_target_router, in_target_area], dim=1)
        x_stage1 = F.relu(self.fc4(_c))
        x_stage1 = F.relu(self.fc64_256(x_stage1))

        _c = torch.cat([boxes_is_found, distance_to_box, in_target_area], dim=1)
        x_stage2 = F.relu(self.fc5(_c))
        x_stage2 = F.relu(self.fc64_256(x_stage2))

        _c = torch.cat([x_img, x_speed, x_distance, x_stage1, x_stage2], dim=1)
        x_concatenated = F.relu(self.fc6(_c))

        x_steer = F.relu(self.fc2(steer))
        x_steer = F.relu(self.fc64_256(x_steer))
        x_steer = F.relu(self.fc7(torch.cat([x_concatenated, x_steer], dim=1)))
        x_steer = F.relu(self.fc256_64(x_steer))

        x_forward = F.relu(self.fc2(forward))
        x_forward = F.relu(self.fc64_256(x_forward))
        x_forward = F.relu(self.fc7(torch.cat([x_concatenated, x_forward], dim=1)))
        x_forward = F.relu(self.fc256_64(x_forward))

        x_target = F.relu(self.fc2(target_found))
        x_target = F.relu(self.fc64_256(x_target))
        x_target = F.relu(self.fc7(torch.cat([x_concatenated, x_target], dim=1)))
        x_target = F.relu(self.fc256_64(x_target))

        _c = torch.cat([x_steer, x_forward, x_target], dim=1)
        x_concatenated2 = F.relu(self.fc8(_c))

        outputs = self.tanh(self.fc9(x_concatenated2))
        return self.max_action * outputs
