from Protos import car_communication_pb2_grpc as CarCommunicationApp_pb2_grpc 
from Protos import car_communication_pb2 as CarCommunicationApp_pb2 
import grpc
import cv2

from random import randint
from config import *

server_address = 'localhost'
server_port = 50051

channel = grpc.insecure_channel(f"{server_address}:{server_port}")
stub = CarCommunicationApp_pb2_grpc.CommunicationStub(channel)


def create_mock_client_request(image_bytes):
    distance_sensors_data = CarCommunicationApp_pb2.DistanceSensorsData( #pyright: ignore
        front_left_distance=1.5,
        front_distance=2.5,
        front_right_distance=1.7,
        back_left_distance=1.8,
        back_distance=-1,
        back_right_distance=float('inf'),
    )

    routers_data = [
        CarCommunicationApp_pb2.RouterData(id='1', rssi=-30.0), #pyright: ignore
        CarCommunicationApp_pb2.RouterData(id='2', rssi=-10.0), #pyright: ignore
        CarCommunicationApp_pb2.RouterData(id='3', rssi=-101.0), #pyright: ignore
    ]

    car_collision_data = randint(0,100) == 3

    client_request = CarCommunicationApp_pb2.ClientRequest( #pyright: ignore
        car_id='A-001',
        camera_image=image_bytes,
        distance_sensors_data=distance_sensors_data,
        routers_data=routers_data,
        boxes_in_camera_view=True,
        car_collision_data=car_collision_data,
        qr_code_metadata='metadata'
    )
    return client_request

def convert_frame_to_bytes(frame) -> bytes:
    encoded_frame = _get_encoded_frame(frame)
    return encoded_frame.tobytes()

def _get_encoded_frame(frame):
    _, encoded_frame = cv2.imencode(TARGET_ENCODE_TO, frame)
    return encoded_frame

def send_video_from_path(path:str) -> None:
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image_bytes = convert_frame_to_bytes(frame)
        send_request(image_bytes)
    cap.release()

def get_command_by_index(index:int) -> str:
    command = CarCommunicationApp_pb2.ServerResponse.Command.Name(index) #pyright: ignore
    return command

def send_request(image_bytes) -> None:
    request = create_mock_client_request(image_bytes)
    response = stub.SendRequest(request)
    command_index = response.command
    command = get_command_by_index(index=command_index)
    print(f'Received command: {command}')


def main():
    path = VIDEO_PATH
    send_video_from_path(path)

if __name__ == '__main__':
    main()

