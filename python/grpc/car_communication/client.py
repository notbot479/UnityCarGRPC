import random
import grpc
import cv2
import os

from Protos import car_communication_pb2
from Protos import car_communication_pb2_grpc


SENSORS_COUNT = 6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURE_PATH = os.path.join(BASE_DIR,'1.jpg')
VIDEO_PATH = os.path.join(BASE_DIR,'1.mp4')
TARGET_ENCODE_TO = '.jpg'

def get_direction_name_by_index(direction_index:int) -> str:
    direction = car_communication_pb2.Command.Direction.Name(direction_index) #pyright: ignore
    return str(direction)

def grpc_send_data(stub, chunk:bytes, sensors_data:list[float,]) -> None:
    fl,f,fr,bl,b,br = sensors_data
    sensors_data_message = car_communication_pb2.SensorsData( #pyright: ignore
            front_left_distance=fl,
                front_distance=f,
                front_right_distance=fr,
                back_left_distance=bl,
                back_distance=b,
                back_right_distance=br,
            )
    request = car_communication_pb2.ClientRequest( #pyright: ignore
        video_frame=chunk,
        sensors_data=sensors_data_message,
    )
    response = stub.SendRequest(request)
    direction_index = response.command.direction
    direction = get_direction_name_by_index(direction_index)
    print(direction)

def get_mock_sensors_data() -> list[float,]:
    return [random.random()*random.randint(1,5) for _ in range(SENSORS_COUNT)]

def send_video_from_path_and_distance(stub,path:str) -> None:
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        chunk = convert_frame_to_bytes(frame)
        sensors_data = get_mock_sensors_data()
        grpc_send_data(stub, chunk, sensors_data)
    cap.release()

def convert_frame_to_bytes(frame) -> bytes:
    encoded_frame = _get_encoded_frame(frame)
    return encoded_frame.tobytes()

def _get_encoded_frame(frame):
    _, encoded_frame = cv2.imencode(TARGET_ENCODE_TO, frame)
    return encoded_frame


def main():
    port = 50051
    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = car_communication_pb2_grpc.CommunicationStub(channel)
    send_video_from_path_and_distance(stub, VIDEO_PATH)


if __name__ == '__main__':
    main()
