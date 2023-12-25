from concurrent import futures
import numpy as np
import random
import grpc
import cv2

from Protos import car_communication_pb2
from Protos import car_communication_pb2_grpc
from config import *

FRAMES = []
CAR_COMMAND_DIRECTIONS = {
        'left':car_communication_pb2.Command.LEFT, #pyright: ignore
        'right':car_communication_pb2.Command.RIGHT, #pyright: ignore
        'up':car_communication_pb2.Command.UP, #pyright: ignore
        'down':car_communication_pb2.Command.DOWN, #pyright: ignore
        'stop':car_communication_pb2.Command.STOP, #pyright: ignore
        }

def display_stream_video():
    global FRAMES
    while True:
        if not(FRAMES): continue
        frame = FRAMES.pop(0)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

def convert_bytes_to_frame(frame_bytes:bytes):
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return frame

class CommunicationServicer(car_communication_pb2_grpc.CommunicationServicer):
    def get_random_direction(self) -> str:
        direction = random.choice(list(CAR_COMMAND_DIRECTIONS.keys()))
        return direction

    def send_response(self,direction:str='stop') -> None:
        ud = CAR_COMMAND_DIRECTIONS.get(direction)
        direction = ud if ud else CAR_COMMAND_DIRECTIONS['stop']
        command = car_communication_pb2.Command(direction=direction) #pyright: ignore
        return car_communication_pb2.ServerResponse(command=command) #pyright: ignore

    def add_frame_to_display(self,frame) -> None:
        global FRAMES
        FRAMES.append(frame)

    def SendRequest(self, request, context): #pyright: ignore
        # receive data from client
        chunk: bytes = request.video_frame
        sensors_data = request.sensors_data
        car_collide_obstacle = int(request.car_collide_obstacle)
        frame = convert_bytes_to_frame(chunk)
        # show video from client
        if SHOW_VIDEO: self.add_frame_to_display(frame)
        # dqn / show received data
        print(f"Processing video chunk: {len(chunk)} bytes")
        print(f"Car collision with obstacle: {car_collide_obstacle}")
        print(sensors_data)
        # return data from server
        direction = self.get_random_direction()
        return self.send_response(direction)

def serve():
    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # processing video
        if SHOW_VIDEO: executor.submit(display_stream_video) 
        # init grpc server
        server = grpc.server(executor)
        car_communication_pb2_grpc.add_CommunicationServicer_to_server(
                CommunicationServicer(), 
                server,
                )
        server.add_insecure_port(f'[::]:{PORT}')
        print(f'Start server on port: {PORT}')
        server.start()
        server.wait_for_termination()

if __name__ == '__main__':
    serve()
