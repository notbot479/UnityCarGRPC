from typing import Generator, Literal
from Protos.car_communication_pb2_grpc import (
    add_CommunicationServicer_to_server,
    CommunicationServicer, #pyright: ignore
)
from Protos.car_communication_pb2 import (
    DistanceSensorsData, #pyright: ignore
    ServerResponse, #pyright: ignore
    ClientRequest, #pyright: ignore
    RouterData, #pyright: ignore
)
from concurrent import futures
import random
import grpc
import cv2

from config import *


def get_command_name_by_index(command_index:int) -> str:
    command = ClientRequest.Command.Name(command_index) 
    return str(command)

def _get_image_bytes() -> bytes:
    with open(PICTURE_PATH, 'rb') as file: return file.read()

def _get_encoded_frame(frame):
    _, encoded_frame = cv2.imencode(TARGET_ENCODE_TO, frame)
    return encoded_frame

def _convert_frame_to_bytes(frame) -> bytes:
    encoded_frame = _get_encoded_frame(frame)
    return encoded_frame.tobytes()

def _get_video_frame_bytes() -> Generator:
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        chunk = _convert_frame_to_bytes(frame)
        yield chunk
    cap.release()

gen = _get_video_frame_bytes()
def get_bytes_data(
    content_type: Literal['video'] | Literal['image'] = 'video',
) -> bytes | None:
    global gen
    if content_type == 'image':
        return _get_image_bytes()
    try:
        return next(gen)
    except StopIteration:
        return None


class CommunicationServicer(CommunicationServicer):
    _grpc_server_response_class = ServerResponse

    def send_response_to_client(
        self,
        car_id:str,
        camera_image:bytes | None,
        boxes_in_camera_view: bool,
        car_collision_data: bool,
        qr_code_metadata: str,
        routers_data:list,
        distance_sensors_data: DistanceSensorsData,
    ):
        response = self._grpc_server_response_class(
            car_id = car_id,
            camera_image = camera_image,
            boxes_in_camera_view = boxes_in_camera_view,
            car_collision_data = car_collision_data,
            qr_code_metadata = qr_code_metadata,
            distance_sensors_data = distance_sensors_data,
            routers_data = routers_data,
        )
        return response

    def _get_mock_distance_sensors_data(self) -> DistanceSensorsData:
        distance_sensors_data = DistanceSensorsData( 
            front_left_distance=1.0,
            front_distance=2.0,
            front_right_distance=3.0,
            back_left_distance=4.0,
            back_distance=5.0,
            back_right_distance=6.0
        )
        return distance_sensors_data

    def _get_mock_routers_data(self, *, count:int=4) -> list:
        routers_data = []
        for router_id in range(1,count+1):
            rid = f'Router{router_id}'
            rssi = - random.randint(0,100)
            router_data = RouterData(id=rid,rssi=rssi)
            routers_data.append(router_data)
        return routers_data

    def SendRequest(self, request, _):
        command_index = int(request.command)
        command = get_command_name_by_index(command_index=command_index)
        print(f"Received command: {command}")
        
        # Mocking server response
        car_id = 'Car1'
        camera_image = get_bytes_data(CAMERA_MODE)
        distance_sensors_data = self._get_mock_distance_sensors_data()
        routers_data = self._get_mock_routers_data()
        boxes_in_camera_view = True
        car_collision_data = False
        qr_code_metadata = "METADATA"
    
        response = self.send_response_to_client(
            car_id = car_id,
            camera_image = camera_image,
            boxes_in_camera_view = boxes_in_camera_view,
            car_collision_data = car_collision_data,
            qr_code_metadata = qr_code_metadata,
            distance_sensors_data = distance_sensors_data,
            routers_data = routers_data,
        )
        return response

def run_server(service, *, max_workers: int = 10, port:int = 50051):
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        server = grpc.server(executor)
        service(CommunicationServicer(),server)
        server.add_insecure_port(f'[::]:{port}')
        print(f'Start server on port: {port}')
        server.start()
        server.wait_for_termination()


def main():
    service = add_CommunicationServicer_to_server
    max_workers = MAX_WORKERS
    port = PORT
    
    run_server(service=service, max_workers=max_workers, port=port)

if __name__ == '__main__':
    main()
