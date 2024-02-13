using System.Collections.Generic;
using UnityEngine;

using System.Threading.Tasks;
using Google.Protobuf;
using Grpc.Core;

using CarCommunicationApp;


public class CommunicationImpl : Communication.CommunicationBase
{
    public override Task<ServerResponse> SendRequest(ClientRequest request, ServerCallContext context)
    {
        string command = $"{request.Command}";

        // receive command from client
        Debug.Log("Received request: " + command);
        // Mock server response
        var response = new ServerResponse
        {
            CarId = "Car1",
            CameraImage = ByteString.CopyFromUtf8("Binary camera data"),
            DistanceSensorsData = new DistanceSensorsData
            {
                FrontLeftDistance = 1.0f,
                FrontDistance = 2.0f,
                FrontRightDistance = 3.0f,
                BackLeftDistance = 4.0f,
                BackDistance = 5.0f,
                BackRightDistance = 6.0f
            },
            BoxesInCameraView = true,
            CarCollisionData = false,
            QrCodeMetadata = "METADATA"
        };

        return Task.FromResult(response);
    }
}

public class CarCommunication : MonoBehaviour
{
    private Server server;
    public int port = 50051;

    void Start()
    {
        server = new Server
        {
            Services = { Communication.BindService(new CommunicationImpl()) },
            Ports = { new ServerPort("localhost", port, ServerCredentials.Insecure) }
        };
        server.Start();
        Debug.Log("Server started on port " + port.ToString());

    }

    void OnDestroy()
    {
        if (server != null)
        {
            server.ShutdownAsync().Wait();
        }
    }
}