from Protos.car_communication_pb2_grpc import (
    CommunicationStub,
)
from Protos.car_communication_pb2 import (
    DistanceSensorsData, #pyright: ignore
    ServerResponse, #pyright: ignore
    ClientRequest, #pyright: ignore
)
import grpc

from dataclasses import dataclass
import numpy as np
import threading
import random
import cv2

from config import *


FRAMES = []
FORCE_STOP_DISPLAY = False
def display_stream_video():
    global FRAMES
    while True:
        if not(FRAMES): continue
        frame = FRAMES.pop(0)
        cv2.imshow('Frame', frame)
        user_break = cv2.waitKey(1) & 0xFF == ord('q')
        if user_break or FORCE_STOP_DISPLAY: break
    cv2.destroyAllWindows()


@dataclass
class DistanceSensorData:
    direction: str
    distance: float

@dataclass
class RouterData:
    id: str
    rssi: float

@dataclass
class GrpcServerResponce:
    car_id: str
    camera_image: np.ndarray | None
    distance_sensors_data: list[DistanceSensorData]
    routers_data: list[RouterData]
    boxes_in_camera_view: bool
    car_collision_data: bool 
    qr_code_metadata: str


class GrpcClient:
    _grpc_client_request_class = ClientRequest
    _grpc_server_response_class = ServerResponse
    _grpc_stub_class = CommunicationStub
    
    # commands based on proto file
    _movement_commands = {
        'left': _grpc_client_request_class.LEFT,
        'right': _grpc_client_request_class.RIGHT,
        'forward': _grpc_client_request_class.FORWARD,
        'backward': _grpc_client_request_class.BACKWARD,
        'stop': _grpc_client_request_class.STOP,
    }
    _extra_commands = {
        'respawn': _grpc_client_request_class.RESPAWN,
        'poweroff': _grpc_client_request_class.POWEROFF,
    }

    _commands = _movement_commands | _extra_commands


    def __init__(self, server_url:str) -> None:
        self.channel = grpc.insecure_channel(server_url)
        self.stub = self._grpc_stub_class(self.channel)

    def get_server_response(self, command:str) -> GrpcServerResponce | None:
        response = self._get_grpc_server_response(command=command)
        if not(response): return
        
        # normalize data
        car_id = str(response.car_id)
        boxes_in_camera_view = bool(response.boxes_in_camera_view)
        car_collision_data = bool(response.car_collision_data)
        qr_code_metadata = str(response.qr_code_metadata)
        camera_image = self._convert_bytes_to_frame(
            response.camera_image,
        )
        distance_sensors_data = self._normalize_sensors_data(
            response.distance_sensors_data,
        )
        routers_data = self._normalize_routers_data(
            response.routers_data,
        )
        
        data = GrpcServerResponce(
            car_id = car_id,
            camera_image = camera_image,
            distance_sensors_data = distance_sensors_data,
            routers_data = routers_data,
            boxes_in_camera_view = boxes_in_camera_view,
            car_collision_data = car_collision_data,
            qr_code_metadata = qr_code_metadata,
        )
        return data

    def get_random_movement_command(self) -> str:
        commands = self.get_available_movement_commands()
        command = random.choice(commands)
        return command

    def get_available_movement_commands(self) -> list:
        commands = list(self._movement_commands.keys())
        return commands

    def _get_grpc_server_response(self, command:str) -> ServerResponse | None:
        grpc_command = self._get_grpc_command(command)
        if grpc_command is None: return
        request = self._grpc_client_request_class(command = grpc_command)
        response = self.stub.SendRequest(request)
        return response

    def _get_grpc_command(self, command:str) -> object | None:
        return self._commands.get(command)

    @staticmethod
    def display_response(response: GrpcServerResponce) -> None:
        print('-'*20)
        print(f'CarId: {response.car_id}')
        print('DistanceSensors:')
        for i in response.distance_sensors_data: print(f'- {i}')
        print('Routers:')
        for i in response.routers_data: print(f'- {i}')
        print(f'BoxesInView: {response.boxes_in_camera_view}')
        print(f'CarCollision: {response.car_collision_data}')
        print(f'QR: {response.qr_code_metadata}')
        print()

    @staticmethod
    def display_frame(frame: np.ndarray | None) -> None:
        FRAMES.append(frame)

    @staticmethod
    def _convert_bytes_to_frame(frame_bytes:bytes) -> np.ndarray | None:
        try:
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            return frame
        except:
            return None

    @staticmethod
    def _normalize_sensors_data(data: DistanceSensorsData) -> list[DistanceSensorData]:
        sensors = (
            ('front_left_distance',data.front_left_distance),
            ('front_distance',data.front_distance),
            ('front_right_distance',data.front_right_distance),
            ('back_left_distance',data.back_left_distance),
            ('back_distance',data.back_distance),
            ('back_right_distance',data.back_right_distance),
        )
        data = []
        for direction, distance in sensors:
            sensor_data = DistanceSensorData(
                direction=str(direction),
                distance=float(distance),
            )
            data.append(sensor_data)
        return data

    @staticmethod
    def _normalize_routers_data(data: list) -> list[RouterData]:
        routers = []
        for d in data:
            router_data = RouterData(id=str(d.id),rssi=float(d.rssi))
            routers.append(router_data)
        return routers


def run_client():
    client = GrpcClient(server_url=SERVER_URL)
    done = False

    #respawn car at first
    current_state = client.get_server_response(command='respawn') #pyright: ignore
    while not done:
        command = client.get_random_movement_command() # simulate dqn
        next_state = client.get_server_response(command=command)
        if not(next_state): continue #TODO BUG?
        if SHOW_VIDEO: client.display_frame(next_state.camera_image)
        if SHOW_DATA: client.display_response(next_state)
        # get reward, done and etc. train dqn

def main():
    global FORCE_STOP_DISPLAY

    if SHOW_VIDEO: threading.Thread(target=display_stream_video).start()
    run_client()
    if SHOW_VIDEO: FORCE_STOP_DISPLAY = True

if __name__ == '__main__':
    main()
