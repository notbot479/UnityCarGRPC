from torch.functional import Tensor
import torch.nn as nn
import torch


class ActorModel(nn.Module):
    def __init__(self) -> None:
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
        self.conv3_img = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.conv4_img = nn.Conv2d(256, 512, kernel_size=2, stride=1)
        self.bn4_img = nn.BatchNorm2d(512)
        self.pool4_img = nn.MaxPool2d(kernel_size=2)
        # img layer 5 (flatten + fc) 
        self.fc_img = nn.Linear(512, 512)
        # Fully connected layers for sensor inputs
        self.fc_sensors1 = nn.Linear(6 + 1 + 1, 128)
        self.fc_sensors2 = nn.Linear(128, 512)
        # Concatenate image and sensor features
        self.fc_combine = nn.Linear(1024, 255)
        # Hint layers
        self.fc_hint1 = nn.Linear(255 + 1, 128)
        self.fc_hint2 = nn.Linear(255 + 1, 128)
        self.fc_hint3 = nn.Linear(255 + 1, 128)
        # Combined hint layer
        self.fc_combined_hint = nn.Linear(128 * 3, 256)
        # Output layer
        self.fc_output = nn.Linear(256, 5)
        
    def forward(
        self, 
        image: Tensor, 
        distance_sensors_distances: Tensor, 
        distance_to_target_router: Tensor,
        distance_to_box: Tensor, 
        in_target_area: Tensor, 
        boxes_is_found: Tensor, 
        target_found: Tensor,
        *args, **kwargs #pyright: ignore
    ) -> Tensor:
        # CNN layers for image input
        x_img = self.pool1_img(nn.functional.relu(self.bn1_img(self.conv1_img(image))))
        x_img = self.pool2_img(nn.functional.relu(self.bn2_img(self.conv2_img(x_img))))
        x_img = self.pool3_img(nn.functional.relu(self.bn3_img(self.conv3_img(x_img))))
        x_img = self.pool4_img(nn.functional.relu(self.bn4_img(self.conv4_img(x_img))))
        x_img = torch.flatten(x_img, 1)
        x_img = nn.functional.relu(self.fc_img(x_img))
        # Fully connected layers for sensor inputs
        x_sensors = torch.cat(
            [
                distance_sensors_distances, 
                distance_to_target_router, 
                distance_to_box
            ], 
            dim=1,
        )
        x_sensors = nn.functional.relu(self.fc_sensors1(x_sensors))
        x_sensors = nn.functional.relu(self.fc_sensors2(x_sensors))
        # Concatenate image and sensor features
        image_and_sensors = torch.cat([x_img, x_sensors], dim=1)
        image_and_sensors = nn.functional.relu(self.fc_combine(image_and_sensors))
        # Hint layers
        x1 = torch.cat([image_and_sensors, in_target_area], dim=1)
        x1 = nn.functional.relu(self.fc_hint1(x1))
        x2 = torch.cat([image_and_sensors, boxes_is_found], dim=1)
        x2 = nn.functional.relu(self.fc_hint2(x2))
        x3 = torch.cat([image_and_sensors, target_found], dim=1)
        x3 = nn.functional.relu(self.fc_hint3(x3))
        # Combined hint layer
        x = torch.cat([x1, x2, x3], dim=1)
        x = nn.functional.relu(self.fc_combined_hint(x))
        # Output layer
        outputs = self.fc_output(x)
        return outputs
