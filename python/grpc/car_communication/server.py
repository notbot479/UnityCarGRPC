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

from dataclasses import dataclass
from concurrent import futures
from collections import deque
from typing import Deque
import numpy as np
import random

from video import (
    convert_bytes_to_frame,
    VideoPlayer, 
)
from config import *


# setting server commands (based on proto file)
CAR_MOVEMENT_SIGNALS = ['noop','left','right','forward','backward','stop']
CAR_EXTRA_SIGNALS = ['poweroff','respawn']


class CameraImage:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
    
    @property
    def bytes_count(self) -> int:
        return len(self.frame.tobytes())

    def __repr__(self) -> str:
        return f"CameraImage [{self.bytes_count} bytes]"

@dataclass
class DistanceSensorData:
    direction: str
    distance: float

@dataclass
class RouterData:
    id: str
    rssi: float

@dataclass
class GrpcClientData:
    car_id: str
    camera_image: CameraImage | None
    distance_sensors: list[DistanceSensorData]
    routers: list[RouterData]
    boxes_in_camera_view: bool
    car_collision_data: bool 
    qr_code_metadata: str


class Servicer(_Servicer):
    @staticmethod
    def generate_grpc_commands(signals:list[str]) -> dict:
        server_response = _Pb2_server_response
        commands = {s:getattr(server_response,s.upper()) for s in signals}
        return commands

    _movement_commands = generate_grpc_commands(CAR_MOVEMENT_SIGNALS)
    _extra_commands = generate_grpc_commands(CAR_EXTRA_SIGNALS)
    _commands = _movement_commands | _extra_commands

    _car_data_deque: Deque[GrpcClientData | None] = deque([None],maxlen=2)
    _car_prev_command: str | None = None


    def __init__(
        self, 
        *args,
        show_stream_video:bool = False,
        show_client_data:bool = False,
        **kwargs,
    ) -> None:
        self.show_stream_video = show_stream_video
        self.show_client_data = show_client_data
        super().__init__(*args,**kwargs)
    
    def processing_client_request(self, data: GrpcClientData):
        if data.car_collision_data: return self._send_respawn_command()
        # load prev data and command
        prev_data = self.get_car_prev_data()
        prev_command = self.get_car_prev_command()
        if not(prev_data and prev_command): return self._send_stop_command()
        # processing data from client
        command = self._get_random_movement()
        return self.send_response_to_client(command)


    @staticmethod
    def get_distance_sensors_distances(
        distance_sensors:list[DistanceSensorData],
    ) -> list[float]:
        distances = [s.distance for s in distance_sensors]
        return distances

    def get_router_rssi_by_id(
        self,
        router_id:str, 
        routers: list[RouterData],
    ) -> float:
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
        if self.show_client_data: self._display_client_data(data)
        if self.show_stream_video and data.camera_image: 
            VideoPlayer.add_frame(data.camera_image.frame)
        grpc_command = self.processing_client_request(data=data)
        if not(grpc_command): return
        return grpc_command


    def _save_car_prev_command(self, command:str) -> None:
        self._car_prev_command = str(command)

    def _save_car_current_data(self, data:GrpcClientData) -> None:
        self._car_data_deque.append(data)

    def _send_stop_command(self):
        return self.send_response_to_client('stop')
    
    def _send_respawn_command(self):
        return self.send_response_to_client('respawn')
    
    def _get_random_movement(self) -> str:
        movement = random.choice(list(self._movement_commands.keys()))
        return movement

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
                distance=round(float(distance), round_factor),
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
                rssi=round(float(d.rssi),round_factor),
            )
            routers.append(router_data)
        return routers

    @staticmethod
    def _display_client_data(data: GrpcClientData) -> None:
        print('-'*20)
        print(f'CarId: {data.car_id}')
        print(f'CameraImage: {data.camera_image}')
        print('DistanceSensors:')
        for i in data.distance_sensors: print(f'- {i}')
        print('Routers:')
        for i in data.routers: print(f'- {i}')
        print(f'BoxesInView: {data.boxes_in_camera_view}')
        print(f'CarCollision: {data.car_collision_data}')
        print(f'QrCodeMetadata: {data.qr_code_metadata}')
        print()

   
def run_server(
    *,
    port:int=50051,
    max_workers:int=10,
    show_stream_video:bool=False,
    show_client_data:bool=False,
) -> None:
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        if show_stream_video: executor.submit(VideoPlayer.display_video)
        server = grpc.server(executor)
        service = Servicer(
            show_stream_video=show_stream_video,
            show_client_data=show_client_data,
        )
        _AddServicer(service,server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        server.wait_for_termination()


if __name__ == '__main__':
    port = PORT
    max_workers = MAX_WORKERS
    show_stream_video = SHOW_STREAM_VIDEO
    show_client_data = SHOW_CLIENT_DATA
    
    print(f'Start server on port: {port}')
    run_server(
        port=port, 
        max_workers=max_workers, 
        show_stream_video=show_stream_video,
        show_client_data=show_client_data,
    )
