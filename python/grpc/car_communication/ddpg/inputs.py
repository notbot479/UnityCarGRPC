import numpy as np
import cv2
import os

from config import AGENT_MODELS_PATH
from .agent import DDPGAgent
from .normalization import *
from units import *


class ModelInputData:
    IMAGE_FIXED_SHAPE: Pixel = 224              
    DISTANCE_SENSOR_MAX_DISTANCE: Meter = 10    
    ROUTER_MAX_RSSI: Rssi = -100                
    CAR_MAX_SPEED: float = 3
    ACTION_DIM:int = 1
    
    DISTANCE_SENSOR_DEFAULT: float = 1        
    ROUTER_DEFAULT: float = 1
    SPEED_DEFAULT: float = 0
    PARAMETER_DEFAULT: float = 0

    def __init__(
        self,
        # car parameters
        speed: float,
        steer: float,
        forward: float,
        backward: float,
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
        # car parameters
        self.speed = self._normalize_speed(speed=speed)
        self.steer = self._normalize_car_parameter(steer)
        self.forward = self._normalize_car_parameter(forward)
        self.backward = self._normalize_car_parameter(backward)
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

    @property
    def inputs(self) -> dict:
        data = {
            # car parameters
            'speed': self.speed,
            'steer': self.steer,
            'forward': self.forward,
            #'backward': self.backward,
            # sensors data
            'image':self.image[:,:,np.newaxis],
            'distance_sensors_distances': self.distance_sensors_distances,
            'distance_to_target_router': self.distance_to_target_router,
            'distance_to_box': self.distance_to_box,
            # hints
            'in_target_area': self.in_target_area,
            'boxes_is_found': self.boxes_is_found,
            'target_found': self.target_found,
        }
        return data

    def _normalize_car_parameter(self, parameter:float) -> float:
        dim = self.ACTION_DIM
        default = self.PARAMETER_DEFAULT
        if not(-dim <= parameter <= dim): return default
        return parameter

    def _normalize_speed(self, speed: float) -> float:
        max_speed = self.CAR_MAX_SPEED
        default = self.SPEED_DEFAULT
        # wtf negative speed
        if not(0 <= speed <= max_speed): return default 
        speed = minmaxscale(speed, 0, max_speed)
        return speed

    def _normalize_image(self, image: np.ndarray | None) -> np.ndarray:
        shape = [self.IMAGE_FIXED_SHAPE, self.IMAGE_FIXED_SHAPE]
        if image is None: # generate random grb image
            image = np.random.randint(0, 256, size=shape+[3], dtype=np.uint8)
        image_shape = image.shape[:-1]
        if image_shape != shape: image = cv2.resize(image, shape)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        normalized_image = grayscale_image.astype(np.float32) / 255
        return normalized_image

    def _normalize_distance(self, distance: Meter) -> float:
        max_distance = self.DISTANCE_SENSOR_MAX_DISTANCE
        default = self.DISTANCE_SENSOR_DEFAULT
        # wtf negative distance
        if not(0 <= distance <= max_distance): return default 
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
        total = '\n== ModelInputData ==\n'
        image_bytes = len(self.image.tobytes())
        # searching target area
        total += f'Image: {image_bytes} bytes\n' 
        total += f'CarParameters:\n'
        d = {
            'speed': self.speed,
            'steer': self.steer,
            'forward': self.forward,
            #'backward': self.backward,
        }
        for k,v in d.items(): total += f'- {k}: {v}\n'
        total += f'Distances: {list(self.distance_sensors_distances)}\n'
        total += f'DistanceToTargetRouter: {self.distance_to_target_router}\n'
        total += f'InTargetArea: {self.in_target_area}\n'
        # searching target box
        total += f'BoxesIsFound: {self.boxes_is_found}\n'
        total += f'DistanceToBox: {self.distance_to_box}\n'
        total += f'TargetIsFound: {self.target_found}\n'
        return total
    

def _test_load_model(agent: DDPGAgent) -> None:
    dir_path = os.path.join(AGENT_MODELS_PATH, 'testmodel')
    agent.load_model(dir_path=dir_path)

def _test_save_model(agent: DDPGAgent) -> None:
    dir_path = os.path.join(AGENT_MODELS_PATH, 'testmodel')
    agent.save_model(dir_path=dir_path)

def _test_train_agent(agent: DDPGAgent, model_input: ModelInputData) -> None:
    # fill reply buffer
    for i in range(agent.reply_buffer.min_capacity):
        state = model_input.inputs
        action = np.array([0.1,] * agent.action_dim)
        reward = float(f'0.{i}')
        next_state = model_input.inputs
        done = False
        agent.reply_buffer.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
    print('\nTrain ddpg agent')
    # train agent
    batch_size = 64
    agent.train(batch_size=batch_size)
    agent.show_stats()
   
def _test_prediction(agent: DDPGAgent, model_input: ModelInputData) -> None:
    actor_network = agent.actor_network
    critic_network = agent.critic_network

    # test predict for ddpg agent
    data = [model_input.inputs,]
    inputs = agent.extract_inputs(data)
    outputs_actor = actor_network.predict(**inputs)
    outputs_critic = critic_network.predict(**inputs, actor_action=outputs_actor)

    print('\n1. Actor and critic outputs:')
    print(f'- Actor: {outputs_actor}')
    print(f'- Critic: {outputs_critic}')
    print('2. Critic extracted qs:')
    critic_qs = agent.extract_qs(outputs_critic)
    print(f'- QS: {critic_qs}')
    
    exploration = True
    qs = agent.get_qs(model_input.inputs, exploration=exploration)
    print('3. DDPG agent qs')
    print(f'- QS: {qs}')

def _test(test_saveload:bool = True):
    agent = DDPGAgent()
    model_input = ModelInputData(
        speed = 2.4,
        steer = 0.5,
        forward = 0.5,
        backward = 0,
        image = None,
        distance_sensors_distances = [1,2,3,4,5,11],
        distance_to_target_router = -101,
        distance_to_box = 8,
        in_target_area = False,
        boxes_is_found = False,
        target_is_found = False,              
    )
    print('\n',model_input)
    # tests
    _test_prediction(agent, model_input)
    _test_train_agent(agent, model_input)
    if test_saveload:
        _test_save_model(agent)
        _test_load_model(agent)

if __name__ == '__main__':
    _test()
