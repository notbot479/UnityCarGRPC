from concurrent import futures
import numpy as np
import grpc
import cv2

from Protos import video_pb2
from Protos import video_pb2_grpc

FRAMES = []


def display_stream_video():
    global FRAMES
    while True:
        if not (FRAMES):
            continue
        frame = FRAMES.pop(0)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def convert_bytes_to_frame(frame_bytes: bytes):
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return frame


class VideoService(video_pb2_grpc.VideoServicer):
    def send_response(self, success: bool = False):
        return video_pb2.VideoFrameResponse(success=success)  # pyright: ignore

    def add_frame_to_display(self, frame) -> None:
        global FRAMES
        FRAMES.append(frame)

    def UploadVideoFrame(self, request, context):  # pyright: ignore
        chunk: bytes = request.chunk
        print(f"Processing video chunk of size: {len(chunk)} bytes")
        frame = convert_bytes_to_frame(chunk)
        self.add_frame_to_display(frame)
        return self.send_response(success=True)


def serve():
    port = 50051
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        # processing video
        executor.submit(display_stream_video)

        # grpc server
        server = grpc.server(executor)
        video_pb2_grpc.add_VideoServicer_to_server(VideoService(), server)
        server.add_insecure_port(f"[::]:{port}")
        print(f"Start server on port: {port}")
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    serve()
