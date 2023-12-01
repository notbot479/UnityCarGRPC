from concurrent import futures
import numpy as np
import grpc
import cv2

from Protos import video_streaming_pb2
from Protos import video_streaming_pb2_grpc

class VideoStreamingServicer(video_streaming_pb2_grpc.VideoStreamingServiceServicer):
    def send_to_client(self,message:str) -> None:
        return video_streaming_pb2.VideoStreamResponse(message=message) #pyright: ignore

    def UploadVideo(self, request_iterator, context): #pyright: ignore
        for request in request_iterator:
            chunk = request.video_chunk
            if not(chunk): self.send_to_client('Failed grab chunk from client')
            video_bytes = request.video_chunk
            frame_array = np.frombuffer(video_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            cv2.imshow('Video from client', frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            print(f"Processing video chunk of size: {len(chunk)} bytes")
        return self.send_to_client("Video processing complete") 

def serve():
    port = 50051
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    video_streaming_pb2_grpc.add_VideoStreamingServiceServicer_to_server(
            VideoStreamingServicer(), 
            server,
            )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started on port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

