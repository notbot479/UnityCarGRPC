import grpc
import cv2
import os

from Protos import video_streaming_pb2
from Protos import video_streaming_pb2_grpc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR,'1.mp4')


def display_video(stub):
    video_path = VIDEO_PATH
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        _, encoded_frame = cv2.imencode('.jpg', frame)
        request = video_streaming_pb2.VideoStreamRequest(video_chunk=encoded_frame.tobytes())
        stub.UploadVideo(iter([request]))
    cap.release()

def run():
    port = 50051
    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = video_streaming_pb2_grpc.VideoStreamingServiceStub(channel)
    display_video(stub)

if __name__ == '__main__':
    run()