from concurrent import futures
from collections import deque
import numpy as np
import random
import grpc
import cv2

from Protos import car_communication_pb2_grpc
from Protos import car_communication_pb2
from logger import logger
from config import *


CAR_COMMAND_DIRECTIONS = {
        'left':car_communication_pb2.Command.LEFT, #pyright: ignore
        'right':car_communication_pb2.Command.RIGHT, #pyright: ignore
        'up':car_communication_pb2.Command.UP, #pyright: ignore
        'down':car_communication_pb2.Command.DOWN, #pyright: ignore
        'stop':car_communication_pb2.Command.STOP, #pyright: ignore
        }
FRAMES = []


def get_random_direction() -> str:
    direction = random.choice(list(CAR_COMMAND_DIRECTIONS.keys()))
    return direction

def add_frame_to_display(frame: np.ndarray) -> None:
    global FRAMES
    FRAMES.append(frame)
    
def display_stream_video() -> None:
    global FRAMES
    while True:
        if not(FRAMES): continue
        frame: np.ndarray = FRAMES.pop(0)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

def convert_bytes_to_frame(frame_bytes:bytes) -> np.ndarray:
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return frame

class CommunicationServicer(car_communication_pb2_grpc.CommunicationServicer):
    def __init__(self,*args,**kwargs) -> None:
        self._car_collision_pool = deque(maxlen=5)
        self._new_round_started = False
        super().__init__(*args,**kwargs)

    def send_response(self,direction:str='stop') -> None:
        ud = CAR_COMMAND_DIRECTIONS.get(direction)
        direction = ud if ud else CAR_COMMAND_DIRECTIONS['stop']
        command = car_communication_pb2.Command(direction=direction) #pyright: ignore
        return car_communication_pb2.ServerResponse(command=command) #pyright: ignore
    
    def get_normalized_sensors_data(
            self, 
            request_sensors_data:car_communication_pb2.SensorsData, #pyright:ignore
        ) -> tuple[float]:
        data = [
            request_sensors_data.front_left_distance,
            request_sensors_data.front_distance,
            request_sensors_data.front_right_distance,
            request_sensors_data.back_left_distance,
            request_sensors_data.back_distance,
            request_sensors_data.back_right_distance,
        ]
        # convert infinity to -1 (relu activation in dqn)
        data = [-1 if float(i) == float('inf') else float(i) for i in data]
        return tuple(data)

    def start_new_round(self, car_collide_obstacle:int) -> bool:
        self._car_collision_pool.append(car_collide_obstacle)
        if car_collide_obstacle:
            if not(self._new_round_started):
                self._new_round_started = True
                return True
            return False
        if self._new_round_started and not(all(self._car_collision_pool)):
            self._new_round_started = False
        return False

    def SendRequest(self, request, context): #pyright: ignore
        # receive data from client / normalize it
        frame = convert_bytes_to_frame(request.video_frame)
        sensors_data = self.get_normalized_sensors_data(request.sensors_data)
        car_collide_obstacle = int(request.car_collide_obstacle)
        if self.start_new_round(car_collide_obstacle):
            logger.warning('TODO dqn start new round')
        # show video from client
        if SHOW_VIDEO: add_frame_to_display(frame)
        # get command from dqn 
        direction = get_random_direction()
        # show log data
        logger.debug(f"Video frame from client: {len(frame.tobytes())} bytes")
        logger.debug(f"Car collision with obstacle: {car_collide_obstacle}")
        logger.debug(f"Data from distance sensors: {sensors_data}")
        logger.debug(f"Predicted direction: {direction}")
        return self.send_response(direction)

def main():
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
        logger.info(f"Start server on port: {PORT}")
        server.start()
        server.wait_for_termination()

if __name__ == '__main__':
    main()
