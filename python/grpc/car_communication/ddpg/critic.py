from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch


class CriticModel(nn.Module):
    def __init__(self, *, action_dim:int):
        super().__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=16, stride=2, padding=7)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=8, stride=2, padding=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(3*3*512, 256)

        # Dense layers
        self.fc2 = nn.Linear(1, 256)  # speed, steer, forward, target
        self.fc3 = nn.Linear(6, 256)  # distance_sensors_distances
        self.fc4 = nn.Linear(2, 256)  # stage 1
        self.fc5 = nn.Linear(3, 256)  # stage 2
        self.fc6 = nn.Linear(action_dim, 256)  # actor action
        self.fc7 = nn.Linear(6*256, 512)  # concatenated
        self.fc8 = nn.Linear(512, 256)  # concatenated
        self.fc9 = nn.Linear(256, 1)  # outputs

    def forward(
        self, 
        # car parameters
        speed: Tensor, 
        # car sensors
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        distance_to_target_router: Tensor,
        distance_to_box: Tensor, 
        # car hints
        in_target_area: Tensor, 
        boxes_is_found: Tensor, 
        #critic input
        actor_action: Tensor, 
        *args,**kwargs #pyright: ignore
    ) -> Tensor:
        # CNN layers
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

        # Dense layers
        x_speed = F.relu(self.fc2(speed))
        x_distance = F.relu(self.fc3(distance_sensors_distances))
        x_stage1 = F.relu(self.fc4(torch.cat([distance_to_target_router, in_target_area], dim=1)))
        x_stage2 = F.relu(self.fc5(torch.cat([boxes_is_found, distance_to_box, in_target_area], dim=1)))
        x_actor = F.relu(self.fc6(actor_action))
        x_concatenated = F.relu(self.fc7(torch.cat([x_img, x_speed, x_distance, x_stage1, x_stage2, x_actor], dim=1)))
        x_concatenated = F.relu(self.fc8(x_concatenated))
        outputs = self.fc9(x_concatenated)
        return outputs
