from Protos.car_communication_pb2_grpc import (
    add_CommunicationServicer_to_server as _AddServicer, #pyright: ignore
    CommunicationServicer as _Servicer, #pyright: ignore
)
from Protos.car_communication_pb2 import (
    ServerResponse as _Pb2_server_response, #pyright: ignore
    ClientRequest as _Pb2_client_request, #pyright: ignore
    DistanceSensorsData as _Pb2_distance_sensors_data, #pyright: ignore
)
import grpc

from dataclasses import dataclass
from concurrent import futures
import numpy as np
import random

from video import (
    convert_bytes_to_frame,
    VideoPlayer, 
)
from config import *


# server commands based on proto file
CAR_MOVEMENT_SIGNALS = ['left','right','forward','backward','stop']
CAR_EXTRA_SIGNALS = ['poweroff','respawn']


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
    camera_image: np.ndarray | None
    distance_sensors_data: list[DistanceSensorData]
    routers_data: list[RouterData]
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
        command = self._get_random_movement()
        return self.send_response_to_client(command)

    def get_grpc_client_data(self, request: _Pb2_client_request) -> GrpcClientData:
        '''parse grpc request and create dataclass'''
        car_id = str(request.car_id)
        boxes_in_camera_view = bool(request.boxes_in_camera_view)
        car_collision_data = bool(request.car_collision_data)
        qr_code_metadata = str(request.qr_code_metadata)
        camera_image = convert_bytes_to_frame(
            request.camera_image,
        )
        distance_sensors_data = self._normalize_sensors_data(
                request.distance_sensors_data,
        )
        routers_data = self._normalize_routers_data(
            request.routers_data,
        )
        data = GrpcClientData(
            car_id = car_id,
            camera_image = camera_image,
            distance_sensors_data = distance_sensors_data,
            routers_data = routers_data,
            boxes_in_camera_view = boxes_in_camera_view,
            car_collision_data = car_collision_data,
            qr_code_metadata = qr_code_metadata,
        )
        return data
    
    def send_response_to_client(self, command:str) -> None:
        grpc_command = self._commands.get(command)
        if grpc_command is None: return
        return _Pb2_server_response(command=grpc_command)
    

    def SendRequest(self, request: _Pb2_client_request, _): 
        data = self.get_grpc_client_data(request) 
        if self.show_client_data: self._display_client_data(data)
        if self.show_stream_video and data.camera_image is not None: 
            VideoPlayer.add_frame(data.camera_image)
        command = self.processing_client_request(data=data)
        if not(command): return
        return command
    
    def _send_respawn_command(self):
        return self.send_response_to_client('respawn')
    
    def _send_poweroff_command(self):
        print('poweroff')
        return self.send_response_to_client('poweroff')

    @staticmethod
    def _normalize_sensors_data(
        data: _Pb2_distance_sensors_data,
    ) -> list[DistanceSensorData]:
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
    def _display_client_data(data: GrpcClientData) -> None:
        print('-'*20)
        print(f'CarId: {data.car_id}')
        if data.camera_image is not None:
            print(f'CameraImage: {len(data.camera_image.tobytes())} bytes')
        print('DistanceSensors:')
        for i in data.distance_sensors_data: print(f'- {i}')
        print('Routers:')
        for i in data.routers_data: print(f'- {i}')
        print(f'BoxesInView: {data.boxes_in_camera_view}')
        print(f'CarCollision: {data.car_collision_data}')
        print(f'QrCodeMetadata: {data.qr_code_metadata}')
        print()

    @staticmethod
    def _normalize_routers_data(data: list) -> list[RouterData]:
        routers = []
        for d in data:
            router_data = RouterData(id=str(d.id),rssi=float(d.rssi))
            routers.append(router_data)
        return routers

    def _get_random_movement(self) -> str:
        movement = random.choice(list(self._movement_commands.keys()))
        return movement

    
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
