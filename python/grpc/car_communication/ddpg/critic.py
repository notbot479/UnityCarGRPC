from torch.functional import Tensor
import torch.nn as nn
import torch


class CriticModel(nn.Module):
    def __init__(self, *, num_actions:int=5):
        super().__init__()
        # img layer 1
        self.conv1_img = nn.Conv2d(1, 64, kernel_size=16, stride=2, padding=0)
        self.bn1_img = nn.BatchNorm2d(64)
        self.pool1_img = nn.MaxPool2d(kernel_size=2)
        # img layer 2
        self.conv2_img = nn.Conv2d(64, 128, kernel_size=8, stride=2, padding=0)
        self.bn2_img = nn.BatchNorm2d(128)
        self.pool2_img = nn.MaxPool2d(kernel_size=2)
        # img layer 3
        self.conv3_img = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.bn3_img = nn.BatchNorm2d(256)
        self.pool3_img = nn.MaxPool2d(kernel_size=2)
        # img layer 4
        self.conv4_img = nn.Conv2d(256, 512, kernel_size=2, stride=1)
        self.bn4_img = nn.BatchNorm2d(512)
        self.pool4_img = nn.MaxPool2d(kernel_size=2)
        # img layer 5 (flatten + fc) 
        self.fc_img = nn.Linear(512, 512)
        # Sensor inputs
        self.fc_sensors = nn.Linear(6 + 2, 512)
        # Action input
        self.fc_action = nn.Linear(num_actions, 512)
        # Concatenate image, sensors, and action
        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(
        self, 
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        distance_to_target_router: Tensor, 
        distance_to_box: Tensor, 
        actor_action: Tensor,
        *args,**kwargs #pyright: ignore
    ) -> Tensor:
        # CNN layers for image input
        x_img = self.pool1_img(nn.functional.relu(self.bn1_img(self.conv1_img(image))))
        x_img = self.pool2_img(nn.functional.relu(self.bn2_img(self.conv2_img(x_img))))
        x_img = self.pool3_img(nn.functional.relu(self.bn3_img(self.conv3_img(x_img))))
        x_img = self.pool4_img(nn.functional.relu(self.bn4_img(self.conv4_img(x_img))))
        x_img = torch.flatten(x_img, 1)
        x_img = nn.functional.relu(self.fc_img(x_img))
        # Sensor processing
        x_sensors = torch.cat(
            (
                distance_sensors_distances, 
                distance_to_target_router, 
                distance_to_box
            ), 
            dim=1,
        )
        x_sensors = torch.relu(self.fc_sensors(x_sensors))
        # Action processing
        x_action = torch.relu(self.fc_action(actor_action))
        # Concatenate image, sensors, and action
        x = torch.cat((x_img, x_sensors, x_action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        outputs = self.fc3(x)
        return outputs
