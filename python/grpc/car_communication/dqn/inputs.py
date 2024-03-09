from typing import Literal, Union
import numpy as np
import cv2

from .agent import DQNAgent, extract_inputs
from units import *


def minmaxscale(
    value:float, 
    min_val:float, 
    max_val:float,
    *,
    round_factor: int = 5
) -> float:
    if min_val == max_val: return 0
    scaled_value = (value - min_val) / (max_val - min_val)
    return round(scaled_value, round_factor)

def zeroOrOne(data:bool) -> Union[Literal[0], Literal[1]]:
    if not(isinstance(data, bool)): return 0
    return 1 if data else 0


class ModelInputData:
    IMAGE_FIXED_SHAPE: Pixel = 64               # small image from camera
    DISTANCE_SENSOR_MAX_DISTANCE: Meter = 10    # ultrasonic max distance
    DISTANCE_SENSOR_DEFAULT:float = -1.0        # relu drop
    ROUTER_MAX_RSSI: Rssi = -100                # typical max rssi 
    ROUTER_DEFAULT: float = -1.0                # relu drop

    def __init__(
        self,
        # car sensors data
        image: np.ndarray | None,
        distance_sensors_distances: list[Meter],
        distance_to_target_router: Rssi,
        distance_to_box: Meter,
        # additional data
        in_target_area: bool,
        boxes_is_found: bool,
        target_is_found: bool,              
    ) -> None:
        # searching target area
        self.image = self._normalize_image(image=image)
        self.distance_sensors_distances = self._normalize_sensors_data(
            distances = distance_sensors_distances,
        )
        self.distance_to_target_router = self._normalize_rssi(
            distance_to_target_router,
        )
        self.in_target_area = zeroOrOne(in_target_area)
        # searching target box
        self.boxes_is_found = zeroOrOne(boxes_is_found)
        self.distance_to_box = self._normalize_distance(
            distance=distance_to_box,
        )
        self.target_found = zeroOrOne(target_is_found)

    def _normalize_image(self, image: np.ndarray | None) -> np.ndarray:
        shape = [self.IMAGE_FIXED_SHAPE, self.IMAGE_FIXED_SHAPE]
        if image is None: return np.zeros(shape=shape)
        image_shape = image.shape[:-1]
        if image_shape != shape: image = cv2.resize(image, shape)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        normalized_image = grayscale_image.astype(np.float32) / 255
        return normalized_image

    def _normalize_distance(self, distance: Meter) -> float:
        max_distance = self.DISTANCE_SENSOR_MAX_DISTANCE
        default = self.DISTANCE_SENSOR_DEFAULT
        # wtf negative distance
        if not(0 < distance < max_distance): return default 
        distance = minmaxscale(distance, 0, max_distance)
        return distance

    def _normalize_sensors_data(self, distances: list[Meter]) -> np.ndarray:
        distances = [self._normalize_distance(i) for i in distances]
        return np.array(distances)

    def _normalize_rssi(self, rssi: Rssi) -> float:
        # abs data to escape relu drop
        rssi = abs(rssi)
        max_rssi = abs(self.ROUTER_MAX_RSSI)
        default = self.ROUTER_DEFAULT
        if rssi > max_rssi: return default
        rssi = minmaxscale(rssi, 0, max_rssi)
        return rssi

    def __repr__(self) -> str:
        total = '== ModelInputData ==\n'
        image_bytes = len(self.image.tobytes())
        # searching target area
        total += f'Image: {image_bytes} bytes\n' 
        total += f'Distances: {self.distance_sensors_distances}\n'
        total += f'DistanceToTargetRouter: {self.distance_to_target_router}\n'
        total += f'InTargetArea: {self.in_target_area}\n'
        # searching target box
        total += f'BoxesIsFound: {self.boxes_is_found}\n'
        total += f'DistanceToBox: {self.distance_to_box}\n'
        total += f'TargetIsFound: {self.target_found}\n'
        return total

   
    @property
    def inputs(self) -> tuple:
        data = (
            self.image[:,:,np.newaxis],
            self.distance_sensors_distances,
            self.distance_to_target_router,
            self.in_target_area,
            self.boxes_is_found,
            self.distance_to_box,
            self.target_found,
        )
        return data


def _test():
    model_input = ModelInputData(
        image = None,
        distance_sensors_distances = [1,2,3,4,5,6],
        distance_to_target_router = 7,
        distance_to_box = 8,
        in_target_area = False,
        boxes_is_found = False,
        target_is_found = False,              
    )
    agent = DQNAgent()
    model = agent.create_model()
    
    # test model predict
    X = extract_inputs([model_input.inputs,])
    qs = model.predict(X)
    print(qs)
    
    # test fit single data
    X = extract_inputs([model_input.inputs,])
    y = np.array(np.random.rand(1, 6))
    y = np.array([np.random.rand(1,6),])
    print([i.shape for i in X])
    model.fit(X,y)
    
    # test fit multiple data
    count = 100
    data = [model_input.inputs] * count
    X = extract_inputs(data)
    y = np.array([np.random.rand(1,6) for _ in range(count)])
    model.fit(X,y)

if __name__ == '__main__':
    _test()
