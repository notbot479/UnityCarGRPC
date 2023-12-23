from concurrent import futures
import numpy as np
import random
import grpc
import cv2

from Protos import car_communication_pb2
from Protos import car_communication_pb2_grpc

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
        chunk: bytes = request.video_frame
        sensors_data = request.sensors_data
        print(f"Processing video chunk of size: {len(chunk)} bytes")
        print(sensors_data)
        frame = convert_bytes_to_frame(chunk)
        self.add_frame_to_display(frame)
        direction = self.get_random_direction()
        return self.send_response(direction)

def serve():
    port = 50051
    max_workers = 24
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # processing video
        executor.submit(display_stream_video) 

        # grpc server
        server = grpc.server(executor)
        car_communication_pb2_grpc.add_CommunicationServicer_to_server(
                CommunicationServicer(), 
                server,
                )
        server.add_insecure_port(f'[::]:{port}')
        print(f'Start server on port: {port}')
        server.start()
        server.wait_for_termination()

if __name__ == '__main__':
    serve()
