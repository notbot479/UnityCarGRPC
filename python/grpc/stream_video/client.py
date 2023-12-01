import grpc
import cv2
import os

from Protos import video_pb2
from Protos import video_pb2_grpc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURE_PATH = os.path.join(BASE_DIR,'1.jpg')
VIDEO_PATH = os.path.join(BASE_DIR,'1.mp4')
TARGET_ENCODE_TO = '.jpg'


def grpc_send_chunk(stub, chunk:bytes) -> None:
    request = video_pb2.VideoFrameRequest(chunk=chunk) #pyright: ignore
    stub.UploadVideoFrame(request)

def send_video_from_path(stub,path:str) -> None:
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        chunk = convert_frame_to_bytes(frame)
        grpc_send_chunk(stub, chunk)
    cap.release()

def send_picture_from_path(stub, path:str) -> None:
    frame = cv2.imread(path) 
    chunk = convert_frame_to_bytes(frame)
    grpc_send_chunk(stub, chunk)

def convert_frame_to_bytes(frame) -> bytes:
    encoded_frame = _get_encoded_frame(frame)
    return encoded_frame.tobytes()

def _get_encoded_frame(frame):
    _, encoded_frame = cv2.imencode(TARGET_ENCODE_TO, frame)
    return encoded_frame


def main():
    port = 50051
    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = video_pb2_grpc.VideoStub(channel)

    #send_picture_from_path(stub, PICTURE_PATH)
    send_video_from_path(stub, VIDEO_PATH)


if __name__ == '__main__':
    main()
