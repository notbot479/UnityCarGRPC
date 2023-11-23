using Grpc.Net.Client;
using GreeterClientApp;
 
using var channel = GrpcChannel.ForAddress("http://localhost:5281");
var client = new Greeter.GreeterClient(channel);
var reply = await client.SayHelloAsync(
    new HelloRequest { Name = "World" }
    );
Console.WriteLine(reply.Message);
