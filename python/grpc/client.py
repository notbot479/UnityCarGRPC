import grpc
from Protos import greet_pb2 as pb2
from Protos import greet_pb2_grpc as pb2_grpc


def run():
    with grpc.insecure_channel('localhost:5281') as channel:
        client = pb2_grpc.GreeterStub(channel)
        response = client.SayHello(pb2.HelloRequest(name='Some'))
    print("Greeter client received: " + response.message)

if __name__ == '__main__':
    run()

