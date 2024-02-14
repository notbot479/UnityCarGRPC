using System.Collections.Generic;
using UnityEngine;
using System;

using System.Threading.Tasks;
using Google.Protobuf;
using Grpc.Core;

using PimDeWitte.UnityMainThreadDispatcher;
using CarCommunicationApp;


public class CommunicationImpl : Communication.CommunicationBase
{
    public string carData = "";
    public string carCommand = "";
    public override Task<ServerResponse> SendRequest(ClientRequest request, ServerCallContext context)
    {
        // receive command from client
        string command = $"{request.Command}";
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            carCommand = command;
        });

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
            QrCodeMetadata = carData,
        };

        return Task.FromResult(response);
    }
}

public class CarCommunication : MonoBehaviour
{
    // server settings
    private CommunicationImpl service;
    private Server server;
    public int port = 50051;
     // car & car data
    private GameObject car;
    private bool carCollideObstacle;
    private string _CarData;
    // router & router data
    private GameObject carRouterReceiver;
    private List<Tuple<string, double>> routersData;
    // sensors & sensors data
    private GameObject RaySensors;
    private Dictionary<string,float> raySensorsData;
    // camera & camera data
    private GameObject CarCamera;
    private byte[] videoFrame;
    private ByteString videoFrameByteString;

    void Start()
    {
        service = new CommunicationImpl();
        server = new Server
        {
            Services = { Communication.BindService(service) },
            Ports = { new ServerPort("localhost", port, ServerCredentials.Insecure) }
        };
        server.Start();
        Debug.Log("Server started on port " + port.ToString());
         // init game objects
        car =  GameObject.Find("Car");
        RaySensors = GameObject.Find("RaySensors");
        CarCamera = GameObject.Find("Camera");
        carRouterReceiver = GameObject.Find("CarRouterReceiver");

    }

    void Update()
    {
        string command = service.carCommand;
        // processing command from grpc server
        if (command == "Respawn") 
        {
            Debug.Log("Respawn command");
        }
        else if (command == "Poweroff") 
        {
            Debug.Log("Poweroff command");
        }
        else // car movement
        {
            car.GetComponent<CarControllerAdvanced>().CarMove(command);
        }
        // collect car data
        service.carData = command;
        // clear old command
        if (command != "") {service.carCommand = "";}

    }
    void OnDestroy()
    {
        if (server != null)
        {
            server.ShutdownAsync().Wait();
            Debug.Log("Server stopped");
        }
    }
}