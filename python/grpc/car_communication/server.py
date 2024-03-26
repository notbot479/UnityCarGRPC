from Protos.car_communication_pb2_grpc import (
    add_CommunicationServicer_to_server as _AddServicer, #pyright: ignore
    CommunicationServicer as _Servicer, #pyright: ignore
)
from Protos.car_communication_pb2 import (
    DistanceSensorsData as _Pb2_distance_sensors_data, #pyright: ignore
    ServerResponse as _Pb2_server_response, #pyright: ignore
    ClientRequest as _Pb2_client_request, #pyright: ignore
)
import grpc

from torch.utils.tensorboard import SummaryWriter
import torch

from functools import cached_property
from typing import Callable, Deque
from dataclasses import dataclass
from concurrent import futures
from collections import deque
from enum import Enum
import threading
import random
import time
import os

from api.web_service import (
    WebServiceRequest,
    WebService,
)
from services.mock.tasks import (
    add_random_mock_task,
    init_mock_tasks, 
)
from services.task_manager import Product
from services.video_manager import (
    convert_bytes_to_frame,
    VideoPlayer, 
)

from ddpg.inputs import ModelInputData
from ddpg.reward import RewardPolicy
from ddpg.agent import DDPGAgent

from client.data import *
from config import *
from units import *


# settings 
TRAIN_AGENT = True
DISABLE_RANDOM = True
CUDA_AVAILABLE = torch.cuda.is_available()
# setting server commands (based on proto file)
CAR_MOVEMENT_SIGNALS = ['left','right','forward','backward','noop']
CAR_EXTRA_SIGNALS = ['poweroff','respawn','stop']

# for more repetitive results
if DISABLE_RANDOM:
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CarActiveTask:
    car_id: str
    product: Product
    route: list[str]

    def __repr__(self) -> str:
        return f'ActiveTask[{self.product}]'

class ServicerMode(Enum):
    READY = 1
    BUSY = 2

class Servicer(_Servicer):
    def __init__(self, *args, **kwargs) -> None:
        init_mock_tasks()
        add_random_mock_task()
        super().__init__(*args,**kwargs)

    @staticmethod
    def busy_until_end(func) -> Callable:
        def lock_servicer_and_execute(*args, **kwargs) -> None:
            Servicer._mode = ServicerMode.BUSY
            func(*args, **kwargs)
            Servicer._mode = ServicerMode.READY

        def wrapper(*args,**kwargs) -> None:
            target = lambda: lock_servicer_and_execute(*args, **kwargs)
            thread = threading.Thread(target=target)
            thread.start()
        return wrapper

    @staticmethod
    def generate_grpc_commands(signals:list[str]) -> dict:
        server_response = _Pb2_server_response
        commands = {s:getattr(server_response,s.upper()) for s in signals}
        return commands

    @staticmethod
    def get_random_qs(num_actions: int) -> np.ndarray:
        qs = np.random.rand(num_actions)
        return qs

    @staticmethod
    def _get_logs_path() -> str:
        tm = int(time.time())
        path = os.path.join(AGENT_LOGS_PATH, str(tm))
        os.makedirs(path)
        return path
    
    # ================================================================================

    epsilon: float = 1
    agent_train_each_step: bool = False
    agent_train_batch_size: int = 64
    agent_train_batches: int = 10
    agent_cuda_train_batch_multiplier: int = 10
    
    # settings: agent
    _agent_episodes_count: int = 5000
    _agent_respawn_very_bad_model: bool = True
    _agent_epsilon_decay:float = 0.99
    _agent_min_epsilon: float = 0.001
    _agent_min_reward: float = -20
    _agent_aggregate_stats_every: int = 10
    # settings: car route
    _car_hit_object_patience = 10
    _car_respawn_nearest_router_id: str = '9'
    # settings: car search target box
    _car_target_patience:int = 5
    _car_ignore_target_area: bool = False
    _target_router_already_locked: bool = False
    # settings: switch router policy
    _car_lock_target_router_rssi: Rssi = -15
    _car_switch_target_router_rssi: Rssi = -10
    _car_switch_target_router_rssi_of_next_shortcut: Rssi = -50
    _car_switch_target_router_rssi_of_next: Rssi = -90
    
    # init service
    show_stream_video = SHOW_STREAM_VIDEO
    show_client_data = SHOW_CLIENT_DATA
    _mode: ServicerMode = ServicerMode.READY
    _web_service = WebService()
    _agent = DDPGAgent(
        load_best_from_dir=AGENT_MODELS_PATH,
    )
    _writer = SummaryWriter(_get_logs_path())
    # server init commands
    _movement_commands = generate_grpc_commands(CAR_MOVEMENT_SIGNALS)
    _extra_commands = generate_grpc_commands(CAR_EXTRA_SIGNALS)
    _commands = _movement_commands | _extra_commands
    # init prev episode memory
    _agent_episode_rewards: list = [_agent_min_reward, ]
    _ctifd_maxlen = _car_target_patience if _car_target_patience > 1 else 1
    _car_target_is_found_deque: Deque[bool] = deque(maxlen=_ctifd_maxlen)
    _car_hit_object_deque: Deque[bool] = deque(maxlen=_car_hit_object_patience)
    _car_target_is_found_state_metadata: str = ""
    _car_data_deque: Deque[GrpcClientData | None] = deque([None],maxlen=2)
    _car_prev_model_input: ModelInputData | None = None
    _car_prev_target_router_id: str | None = None
    _car_active_task: CarActiveTask | None = None
    _car_prev_command: str | None = None
    _agent_num_actions = len(_movement_commands)
    _agent_prev_qs: np.ndarray = get_random_qs(_agent_num_actions)
    # init counters and flags
    _agent_episode_total_score: Score = 0
    _agent_episode_id: int = 0
    _agent_state_id: int = 1

    # ================================================================================
    
    @busy_until_end
    def agent_start_training(self, car_id: str, routers: list[RouterData]) -> None:
        nearest_router = self.get_nearest_router(routers)
        if not(nearest_router):
            print('Failed get task. No routers around car.')
            return
        # rewrite car respawn nearest router
        self._car_respawn_nearest_router_id = nearest_router.id
        # create request to web server and set active task for car
        status = self.set_active_task_for_car(
            car_id=car_id, 
            nearest_router_id=nearest_router.id,
        )
        print(status.msg)
        if not(status.ok): return
        self.start_new_episode()
        
    @busy_until_end
    def agent_end_episode(self, data: GrpcClientData) -> None:
        # show some stats
        print(f'== END episode[{self.episode_id}] state[{self.state_id}] ==')
        print(f'- epsilon [{self.epsilon}]')
        print(f'- total score [{self.episode_total_score}]')
        print()
        # train agent on episode end
        if TRAIN_AGENT:
            batches_count = self._agent_train_each_episode_batches
            if not(self.agent_train_each_step) and batches_count:
                self._agent.train_on_episode_end(
                    batch_size = self.agent_train_batch_size,
                    batches_count = batches_count,
                )
        # get task (new task) from respawn router (training start nearest router)
        respawn_router_id = self._car_respawn_nearest_router_id
        self.set_active_task_for_car(
            car_id=data.car_id,
            nearest_router_id=respawn_router_id,
        )
        # save statictic and save model
        round_factor = 3
        self._agent_episode_rewards.append(self.episode_total_score)
        _a = self.episode_id == 1
        _b = not(self.episode_id % self._agent_aggregate_stats_every) 
        if _a or _b:
            # calculate reward stats
            aggregate_every = self._agent_aggregate_stats_every
            ep_batch: list[float] = self._agent_episode_rewards[-aggregate_every:]
            average_reward = round(sum(ep_batch)/len(ep_batch),round_factor)
            min_reward = round(min(ep_batch),round_factor) 
            max_reward = round(max(ep_batch),round_factor)
            reward_data = {
                'min_reward':min_reward, 
                'max_reward':max_reward,
                'average_reward':average_reward,
            }
            # tensorboard update stats (write logs)
            extra_stats = {'epsilon': self.epsilon}
            stats = self._agent.stats | reward_data | extra_stats
            for k,v in stats.items():
                self._writer.add_scalar(k,v,self.episode_id)
            # Save model
            if not(_a):
                dir_path = self.get_agent_save_dir_path(data=reward_data)
                self._agent.save_model(dir_path=dir_path)
        # Decay epsilon
        if self.epsilon > self._agent_min_epsilon:
            self.epsilon *= self._agent_epsilon_decay
            self.epsilon = max(self._agent_min_epsilon, self.epsilon)
        self.start_new_episode()

    @busy_until_end
    def router_controller_update_route(
        self,
        car_id:str,
        routers: list[RouterData],
    ) -> None:
        current_target_router_id = self.get_car_target_router_id()
        next_target_router_id = self.get_car_next_target_router_id()
        if not(current_target_router_id): return
        current_target_rssi = self.get_router_rssi_by_id(
            router_id=current_target_router_id,
            routers=routers,
        )
        current_target_rssi = current_target_rssi
        next_router_rssi = float('inf')
        if next_target_router_id is not None:
            next_router_rssi = self.get_router_rssi_by_id(
                router_id=next_target_router_id,
                routers=routers,
            )
        # car lose connection with target router, get new route based on nearest router
        if current_target_rssi == float('inf'):
            nearest_router = self.get_nearest_router(routers=routers)
            if not(nearest_router):
                print('TODO no routers in car view. DO some')
                return
            self.set_active_task_for_car(
                car_id=car_id,
                nearest_router_id=nearest_router.id,
            )
            return
        # switch target router if current has `nice` rssi
        g_rssi = self._car_switch_target_router_rssi
        g_rssi_next = self._car_switch_target_router_rssi_of_next
        _a = abs(current_target_rssi) < abs(g_rssi) 
        _b = abs(next_router_rssi) < abs(g_rssi_next)
        if _a and _b:
            self.car_switch_target_router()
            return
        # if next target router has `good` rssi -> switch target router
        s_rssi_next = self._car_switch_target_router_rssi_of_next_shortcut
        if abs(next_router_rssi) < abs(s_rssi_next):
            self.car_switch_target_router()
            return
    
    def processing_client_request(self, data: GrpcClientData):
        # send poweroff command, if max train episodes reached 
        if self.episode_id > self.max_train_episodes:
            return self._send_poweroff_command()
        # skip car data if server current busy
        if self.mode == ServicerMode.BUSY: 
            return self._send_stop_command() 
        # load prev data and command
        prev_data = self.get_car_prev_data()
        prev_command = self.get_car_prev_command()
        prev_target_router_id = self.get_car_prev_target_router_id() 
        prev_model_input = self.get_prev_model_input()
        # start agent training, get active task from web service
        if not(prev_data and prev_command):
            self.agent_start_training(car_id=data.car_id, routers=data.routers)
            return self._send_stop_command()
        # at start: do nothing, if no active task or target router not visible 
        if not(self.car_active_task and prev_target_router_id): 
            return self._send_stop_command()
        # update active task route based on policy
        self.router_controller_update_route(car_id=data.car_id, routers=data.routers) 
        # normalize data for nerual networks: range (0 to 1 or -1 to 1)
        model_input = self.get_model_input_data(data=data)
        if not(model_input and prev_model_input): 
            return self._send_stop_command()
        # calculate reward and get done based on policy
        reward, done = self.get_reward_and_done(prev_data, data)
        self.total_score_add_reward(reward) 
        # show some stats
        print(f'Route: {self.car_active_task.route}')
        print(f'Reward: {reward}\n')
        # train agent
        if TRAIN_AGENT:
            self._agent.reply_buffer.store(
                state=prev_model_input.inputs,
                action=self._agent_prev_qs,
                reward=reward,
                next_state=model_input.inputs,
                done=bool(done),
            )
            if self.agent_train_each_step:
                self._agent.train(
                    terminal_state=bool(done),
                    batch_size=self.agent_train_batch_size,
                )
        # end episode and respawn very bad model (reached min reward)
        if self.respawn_very_bad_model():
            self.agent_end_episode(data=data)
            return self._send_respawn_command()
        # agent end episode based on train policy
        if done == Done.TARGET_IS_FOUND:
            print('Send message to web - task complate')
            add_random_mock_task()
            self.agent_end_episode(data=data)
            return self._send_respawn_command()
        elif done == Done.HIT_OBJECT:
            self.agent_end_episode(data=data)
            return self._send_respawn_command()
        # agent predict command or get random movement
        if np.random.random() > self.epsilon:
            qs = self._agent.get_qs(model_input.inputs)
            self._agent_prev_qs = qs
            movement_index = int(np.argmax(qs))
            command = self.get_movement_by_index(movement_index)
            print(f'Send predicted command to client: {command}')
        else:
            rand_qs = self.get_random_qs(self._agent_num_actions)
            self._agent_prev_qs = rand_qs
            movement_index = int(np.argmax(rand_qs))
            command = self.get_movement_by_index(movement_index)
            print(f'Send random command to client: {command}')
        return self.send_response_to_client(command)

    # ===============================================================================
    
    def respawn_very_bad_model(self) -> bool:
        if not(self._agent_respawn_very_bad_model): return False
        return self.episode_total_score < self._agent_min_reward

    def get_agent_save_dir_path(self, *, data: dict = {}) -> str:
        tm = int(time.time())
        path = AGENT_MODELS_PATH
        name = '_'.join([f'{k}[{v}]' for k,v in data.items()])
        model_name = f'model_{name}_{tm}'
        model_path = os.path.join(path, model_name)
        return model_path

    def get_random_movement(self) -> str:
        movements = self._get_movement_list()
        movement = random.choice(movements)
        return movement

    def get_movement_index(self, movement: str) -> int | None:
        movements = self._get_movement_list()
        try:
            ind = movements.index(movement)
            return ind
        except:
            return None

    def get_movement_by_index(self, index:int, *, default:str='noop') -> str:
        movements = self._get_movement_list()
        try:
            movement = movements[index]
            return movement
        except:
            return default

    def get_prev_model_input(self) -> ModelInputData | None:
        return self._car_prev_model_input

    @property
    def min_epsilon(self) -> float:
        return self._agent_min_epsilon

    @property
    def epsilon_decay(self) -> float:
        return self._agent_epsilon_decay
 
    @property
    def max_train_episodes(self) -> int:
        return self._agent_episodes_count

    @property
    def car_hit_object_end_patience(self) -> bool:
        all_hits = all(self._car_hit_object_deque)
        filled = len(self._car_hit_object_deque) == self._car_hit_object_deque.maxlen
        return all_hits and filled

    def get_reward_and_done(
        self, 
        old_state: GrpcClientData,
        new_state: GrpcClientData,
        *,
        done: Done = Done._,
    ) -> tuple[Score, Done]:
        # get target found based on patience
        car_hit_object = new_state.car_collision_data
        in_target_area = self.car_in_target_area(new_state.routers)
        if self._car_target_patience > 1:
            target_found = self.is_target_found_and_locked(
                qr_metadata=new_state.qr_code_metadata,
            )
        else:
            target_found = self.is_target_box_qr(
                qr_metadata=new_state.qr_code_metadata,
            )
        # add car hit to patience
        if self._car_hit_object_patience: 
            self._car_hit_object_deque.append(car_hit_object)
        # done policy
        if self._car_hit_object_patience and self.car_hit_object_end_patience:
            done = Done.HIT_OBJECT
            self._car_hit_object_deque.clear()
        elif target_found and in_target_area: 
            done = Done.TARGET_IS_FOUND
        # reward policy
        target_router_id = self.get_car_target_router_id()
        target_router_switched = self.is_target_router_switched
        if car_hit_object:
            reward = RewardPolicy.HIT_OBJECT.value
            return (reward, done)
        if not(target_router_id): 
            reward = RewardPolicy.PASSIVE_REWARD.value
            return (reward, done)
        if not(in_target_area): 
            # stage 1: route
            old_target_rssi = self.get_router_rssi_by_id(
                router_id=target_router_id,
                routers=old_state.routers,
            )
            new_target_rssi = self.get_router_rssi_by_id(
                router_id=target_router_id,
                routers=new_state.routers,
            )
            delta = round(old_target_rssi - new_target_rssi, 1)
            if delta == 0:
                reward = RewardPolicy.PASSIVE_REWARD.value
                return (reward, done)
            if target_router_switched or delta < 0:
                reward = RewardPolicy.SHORTEN_DISTANCE_TO_ROUTER.value
                return (reward, done)
            else:
                reward = RewardPolicy.INCREASE_DISTANCE_TO_ROUTER.value
                return (reward, done)
        else: 
            # stage 2: search
            boxes_in_view = new_state.boxes_in_camera_view
            target_is_found = self.is_target_box_qr(
                qr_metadata=new_state.qr_code_metadata,
            )
            if not(boxes_in_view):
                reward = RewardPolicy.IN_TARGET_AREA_NO_BOXES_FOUND.value
                return (reward, done)
            if target_is_found:
                reward = RewardPolicy.TARGET_IS_FOUND.value
                return (reward, done)
            # get distance from front sensor
            old_front_sensor = self.get_distance_sensor_by_direction(
                direction='front',
                distance_sensors=old_state.distance_sensors,
            )
            new_front_sensor = self.get_distance_sensor_by_direction(
                direction='front',
                distance_sensors=new_state.distance_sensors,
            )
            if not(old_front_sensor and new_front_sensor):
                reward = RewardPolicy.PASSIVE_REWARD.value
                return (reward, done)
            delta = abs(old_front_sensor.distance) - abs(new_front_sensor.distance)
            delta = round(delta, 2)
            if boxes_in_view and delta == 0:
                reward = RewardPolicy.IN_TARGET_AREA_BOXES_FOUND.value
                return (reward, done)
            elif delta == 0:
                reward = RewardPolicy.PASSIVE_REWARD.value
                return (reward, done)
            if delta > 0:
                reward = RewardPolicy.SHORTEN_DISTANCE_TO_BOX.value
                return (reward, done)
            else:
                reward = RewardPolicy.INCREASE_DISTANCE_TO_BOX.value
                return (reward, done)
    
    def car_switch_target_router(self) -> None:
        active_task = self.car_active_task
        if not(active_task): return
        if len(active_task.route) < 2: return
        active_task.route.pop(0)

    @property
    def is_target_router_switched(self) -> bool: 
        prev_router_id = self.get_car_prev_target_router_id()
        current_router_id = self.get_car_target_router_id()
        return prev_router_id != current_router_id

    def set_active_task_for_car(
        self, 
        car_id: str, 
        nearest_router_id: str
    ) -> Status:
        # get active task from web service
        product, route = self.get_car_active_task(
            car_id = car_id,
            nearest_router_id = nearest_router_id,
        )
        # set active task for car, if ok response
        if not(product): 
            msg = f'No active task for car with id: {car_id}'
            return Status(ok=False,msg=msg)
        if not(route):
            msg = f'Failed get route to target router, nearest: {nearest_router_id}'
            return Status(ok=False,msg=msg)
        active_task = CarActiveTask(
            car_id=car_id,
            product=product, 
            route=route,
        )
        self._car_active_task = active_task
        msg = f'Success: {active_task}; CarID: {car_id}'
        return Status(ok=True, msg=msg)

    def lock_target_area(self, router_rssi: Rssi) -> bool:
        if self._target_router_already_locked: return True
        good_rssi = abs(router_rssi) < abs(self._car_lock_target_router_rssi)
        if not(good_rssi): return False
        self._target_router_already_locked = True
        return True

    def car_in_target_area(self, routers: list[RouterData]) -> bool:
        active_task = self.car_active_task
        if not(active_task and active_task.route): return False        
        last_router_in_route = len(active_task.route) == 1
        if not(last_router_in_route): 
            self._target_router_already_locked = False
            return False
        target_router = self.get_router_by_id(
            router_id=active_task.route[0],
            routers=routers,
        )
        if not(target_router): return False
        target_area_locked = self.lock_target_area(router_rssi=target_router.rssi)
        return last_router_in_route and target_area_locked

    def is_target_found_and_locked(self, qr_metadata: str) -> bool:
        '''use is_target_box_qr at first or send qr_metadata'''
        self.is_target_box_qr(qr_metadata=qr_metadata)
        d = self._car_target_is_found_deque
        found = all(d) and len(d) == d.maxlen
        # clear deque if found
        if found: d.clear()
        return found

    def is_target_box_qr(self, qr_metadata: str) -> bool:
        active_task = self.car_active_task
        if not(active_task): return False
        target_qr = active_task.product.qr_code_metadata 
        result = target_qr == qr_metadata
        # add data to deque, used in is_target_found_and_locked
        metadata = f'{self._agent_episode_id}:{self._agent_state_id}'
        if metadata != self._car_target_is_found_state_metadata:
            self._car_target_is_found_deque.append(result)
            self._car_target_is_found_state_metadata = metadata
        return result

    def get_model_input_data(self, data: GrpcClientData) -> ModelInputData | None:
        target_router_id = self.get_car_target_router_id()
        front_sensor = self.get_distance_sensor_by_direction(
            direction = 'front',
            distance_sensors = data.distance_sensors,
        )
        if not(target_router_id and front_sensor): return
        # first part data
        image = data.camera_image.frame if data.camera_image else None
        distance_sensors_distances = [i.distance for i in data.distance_sensors]
        distance_to_target_router = self.get_router_rssi_by_id(
            router_id = target_router_id,
            routers = data.routers,
        )
        if not(self._car_ignore_target_area):
            in_target_area = self.car_in_target_area(routers=data.routers)
        else:
            in_target_area = True
        # second part data
        if in_target_area:
            boxes_is_found = data.boxes_in_camera_view
            distance_to_box = front_sensor.distance if boxes_is_found else float('inf')
            target_found = self.is_target_box_qr(data.qr_code_metadata)
        else:
            distance_to_box = float('inf')
            boxes_is_found = False
            target_found = False
        # convert to model input data
        model_input_data = ModelInputData(
            image=image,
            distance_sensors_distances = distance_sensors_distances,
            distance_to_target_router = distance_to_target_router,
            distance_to_box = distance_to_box,
            in_target_area = in_target_area,
            boxes_is_found = boxes_is_found,
            target_is_found = target_found,
            )
        # save prev model input data
        self._save_model_input(model_input=model_input_data)
        return model_input_data

    def start_new_episode(self) -> None: 
        # reset env counters
        self._agent_episode_id += 1
        self.reset_total_score()
        self.clear_state()

    def clear_episode(self) -> None:
        '''reset episode env data'''
        self._agent_episode_id = 1
        self.reset_total_score()
        self.clear_state()

    def clear_state(self) -> None:
        self._agent_state_id = 1

    def reset_total_score(self) -> None:
        self._agent_episode_total_score = 0

    def total_score_add_reward(
        self, 
        reward: Score = 0,
        *,
        round_factor: int = 3,
    ) -> None:
        '''add positive or negative reward to total score'''
        reward = self._normalize_reward(reward, round_factor=round_factor)
        self._agent_episode_total_score += reward

    def get_car_target_router_id(self) -> str | None:
        if not(self.car_active_task): return
        route = self.car_active_task.route
        if len(route) == 0: return None
        return self.car_active_task.route[0]

    def get_car_next_target_router_id(self) -> str | None:
        if not(self.car_active_task): return
        route = self.car_active_task.route
        if len(route) < 2: return None
        return self.car_active_task.route[1]
    
    def get_car_prev_target_router_id(self) -> str | None:
        return self._car_prev_target_router_id

    def get_car_active_task(
        self, 
        car_id: str, 
        nearest_router_id:str,
    ) -> tuple[Product | None, list[str]]:
        request = WebServiceRequest(
            car_id=car_id,
            nearest_router_id=nearest_router_id,
        )
        response = self._web_service.send_request(request)
        product, route = response.product, response.route
        return product, route

    @property
    def state_id(self) -> int:
        return self._agent_state_id

    @property
    def episode_id(self) -> int:
        return self._agent_episode_id

    @property
    def episode_total_score(self) -> Score:
        return self._agent_episode_total_score

    @property
    def car_active_task(self) -> CarActiveTask | None:
        return self._car_active_task

    @property
    def mode(self) -> ServicerMode: 
        return self._mode

    def get_router_rssi_by_id(
        self,
        router_id: str, 
        routers: list[RouterData],
    ) -> Rssi:
        router = self.get_router_by_id(router_id=router_id, routers=routers)
        if router is None: return float('inf')
        rssi = router.rssi
        return rssi

    @staticmethod
    def get_distance_sensor_by_direction(
        direction:str, 
        distance_sensors: list[DistanceSensorData]
    ) -> DistanceSensorData | None:
        for distance_sensor in distance_sensors:
            if str(distance_sensor.direction) == str(direction):
                return distance_sensor
        return None

    def get_nearest_router(self, routers: list[RouterData]) -> RouterData | None:
        if len(routers) == 0: return None
        if len(routers) == 1: return routers[0]
        router_id, _ = min([(i.id,abs(i.rssi)) for i in routers], key=lambda x: x[1])
        router = self.get_router_by_id(router_id=router_id,routers=routers)
        return router

    @staticmethod
    def get_router_by_id(
        router_id:str, 
        routers: list[RouterData],
    ) -> RouterData | None:
        for router in routers:
            if str(router.id) == str(router_id):
                return router
        return None

    def get_car_prev_command(self) -> str | None:
        return self._car_prev_command
   
    def get_car_prev_data(self) -> GrpcClientData | None:
        data = self._car_data_deque[0]
        return data

    # ================================================================================
    
    def get_grpc_client_data(self, request: _Pb2_client_request) -> GrpcClientData:
        '''parse grpc request and create dataclass'''
        car_id = str(request.car_id)
        boxes_in_camera_view = bool(request.boxes_in_camera_view)
        car_collision_data = bool(request.car_collision_data)
        qr_code_metadata = str(request.qr_code_metadata)
        frame = convert_bytes_to_frame(request.camera_image)
        camera_image = CameraImage(frame) if frame is not None else None
        distance_sensors = self._normalize_distance_sensors_data(
            request.distance_sensors_data,
        )
        routers = self._normalize_routers_data(request.routers_data)
        data = GrpcClientData(
            car_id = car_id,
            camera_image = camera_image,
            distance_sensors = distance_sensors,
            routers = routers,
            boxes_in_camera_view = boxes_in_camera_view,
            car_collision_data = car_collision_data,
            qr_code_metadata = qr_code_metadata,
        )
        return data
    
    def send_response_to_client(self, command:str) -> None: 
        grpc_command = self._commands.get(command)
        if grpc_command is None: return
        # save server command, if grpc_command exists
        self._save_car_prev_command(command)
        return _Pb2_server_response(command=grpc_command)
 
    def SendRequest(self, request: _Pb2_client_request, _): 
        data = self.get_grpc_client_data(request)
        self._save_car_current_data(data)
        if self.show_client_data: print(data) 
        if self.show_stream_video and data.camera_image: 
            VideoPlayer.add_frame(data.camera_image.frame)
        grpc_command = self.processing_client_request(data=data)
        self._car_prev_target_router_id = self.get_car_target_router_id()
        # processing send grpc command to client
        if not(grpc_command): return
        self._agent_state_id += 1
        return grpc_command

    # ================================================================================

    @cached_property
    def _agent_train_each_episode_batches(self) -> int:
        if not(CUDA_AVAILABLE): return self.agent_train_batches
        batches_cuda = self.agent_cuda_train_batch_multiplier*self.agent_train_batches
        return batches_cuda

    def _get_movement_list(self) -> list:
        return CAR_MOVEMENT_SIGNALS

    def _save_model_input(self, model_input: ModelInputData | None) -> None:
        self._car_prev_model_input = model_input

    @staticmethod
    def _normalize_reward(reward: Score, round_factor: int) -> float:
        reward = round(float(reward), round_factor)
        return reward

    def _save_car_prev_command(self, command:str) -> None:
        self._car_prev_command = str(command)

    def _save_car_current_data(self, data:GrpcClientData) -> None:
        self._car_data_deque.append(data)

    def _send_poweroff_command(self):
        return self.send_response_to_client('poweroff')

    def _send_stop_command(self):
        return self.send_response_to_client('stop')
    
    def _send_respawn_command(self):
        return self.send_response_to_client('respawn')
    
    @staticmethod
    def _normalize_distance_sensors_data(
        data: _Pb2_distance_sensors_data,
        *,
        round_factor:int = 5,
    ) -> list[DistanceSensorData]:
        sensors = (
            ('front_left',data.front_left_distance),
            ('front',data.front_distance),
            ('front_right',data.front_right_distance),
            ('back_left',data.back_left_distance),
            ('back',data.back_distance),
            ('back_right',data.back_right_distance),
        )
        data = []
        for direction, distance in sensors:
            sensor_data = DistanceSensorData(
                direction=str(direction),
                distance=round(Meter(distance), round_factor),
            )
            data.append(sensor_data)
        return data

    @staticmethod
    def _normalize_routers_data(
        data: list,
        *,
        round_factor:int = 5,
    ) -> list[RouterData]:
        routers = []
        for d in data:
            router_data = RouterData(
                id=str(d.id),
                rssi=round(Rssi(d.rssi),round_factor),
            )
            routers.append(router_data)
        return routers

 
def run_server(*, port:int = 50051, max_workers:int = 10) -> None:
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor: 
        server = grpc.server(executor)
        service = Servicer()
        # show stream video
        if service.show_stream_video:
            executor.submit(VideoPlayer.display_video)
        # init grpc server
        _AddServicer(service,server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        server.wait_for_termination()

if __name__ == '__main__':
    port = PORT
    max_workers = MAX_WORKERS
    print(f'[Service] Start server on port: {port}')
    run_server(port=port, max_workers=max_workers)
