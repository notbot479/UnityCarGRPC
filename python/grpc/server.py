from concurrent import futures
from Protos import greet_pb2 as pb2
from Protos import greet_pb2_grpc as pb2_grpc

import grpc

class GreeterServicer(pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return pb2.HelloReply(message=f'Hi!')

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_GreeterServicer_to_server(GreeterServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started. Listening on port 50051.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

