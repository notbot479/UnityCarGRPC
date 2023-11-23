using Grpc.Core;
using GreeterServiceApp;

namespace GreeterServiceApp.Services;

public class GreeterService : Greeter.GreeterBase
{
    private readonly ILogger<GreeterService> _logger;
    public GreeterService(ILogger<GreeterService> logger)
    {
        _logger = logger;
    }
    public override Task<HelloReply> SayHello(HelloRequest request, ServerCallContext context)
        {
            _logger.LogInformation($"Received request from user: {request.Name}");

            var reply = new HelloReply
            {
                Message = "Hello " + request.Name
            };
            return Task.FromResult(reply);
        }
}
