using System.Collections.Generic;
using System;

using UnityEngine;

using Cysharp.Net.Http;
using Grpc.Net.Client;
using Grpc.Core;

using Google.Protobuf;
using CarCommunicationApp;


public class CarCommunication : MonoBehaviour
{
    public string carID = "1";
    public string serverApiUrl = "http://localhost:50051";
    public bool sendRequestToServer = true;
    public bool moveCarByAI = true;
   
    // grpc client and server
    private GrpcChannel channel;
    private Communication.CommunicationClient client;
    private bool serverConnectionError = false;
    // car & car data
    private GameObject car;
    private bool carCollideObstacle;
    private string command;
    // router & router data
    private GameObject carRouterReceiver;
    private List<Tuple<string, float>> routersData;
    // sensors & sensors data
    private GameObject raySensors;
    private Dictionary<string,float> raySensorsData;
    // camera & camera data
    private GameObject carCamera;
    private byte[] videoFrame;
    private ByteString videoFrameByteString;

    public void Start()
    {
        // create grpc channel and client
        if (sendRequestToServer){
            channel = GrpcChannel.ForAddress(serverApiUrl, new GrpcChannelOptions
            {
                HttpHandler = new YetAnotherHttpHandler { Http2Only = true },
                DisposeHttpClient = true
            });
            client = new Communication.CommunicationClient(channel);
        }
        // init game objects
        car =  GameObject.Find("Car");
        raySensors = GameObject.Find("RaySensors");
        carCamera = GameObject.Find("Camera");
        carRouterReceiver = GameObject.Find("CarRouterReceiver");
    }

    public void Update()
    {
        // get data from sensors, routers, camera
        videoFrame = carCamera.GetComponent<CameraData>().getFrameInBytes();
        videoFrameByteString = ByteString.CopyFrom(videoFrame);
        raySensorsData = raySensors.GetComponent<RaySensorsData>().GetSensorsData();
        routersData = carRouterReceiver.GetComponent<CarRouterReceiver>().GetRoutersData();
        carCollideObstacle = car.GetComponent<CarCollisionData>().isCollide;
        // create grpc request
        if (!sendRequestToServer) { return; }
        var request = new ClientRequest
        {
            CarId = carID,
            CameraImage = videoFrameByteString,
            DistanceSensorsData = new DistanceSensorsData
            {
                FrontLeftDistance = raySensorsData["FrontLeftDistance"],
                FrontDistance = raySensorsData["FrontDistance"],
                FrontRightDistance = raySensorsData["FrontRightDistance"],
                BackLeftDistance = raySensorsData["BackLeftDistance"],
                BackDistance = raySensorsData["BackDistance"],
                BackRightDistance = raySensorsData["BackRightDistance"],
            },
            CarCollisionData = carCollideObstacle,
            // TODO not implemented, send mock data
            BoxesInCameraView = false,
            QrCodeMetadata = "metadata",
        };
        // add repeated routers data to request
        foreach (var t in routersData)
        {
            request.RoutersData.Add(new RouterData { Id = t.Item1, Rssi = t.Item2 });
        }
        // send data using grpc and receive command from server
        try
        {
            var response = client.SendRequest(request);
            command = response.Command.ToString();
            serverConnectionError = false;
        }
        catch
        {
            if (!serverConnectionError)
            {
                Debug.Log("Failed connect to server by url: " + serverApiUrl);
                serverConnectionError = true;
            }
            return;
        }
        // processing command from server
        if (!moveCarByAI) { return; }
        try{
            // TODO respawn command
            // TODO poweroff command
            car.GetComponent<CarControllerAdvanced>().CarMove(command);
        }
        catch{
            Debug.Log("Simulation is stopped. Ignore command from server");
        }
    }

    private void OnDestroy()
    {
        channel?.ShutdownAsync().Wait();
    }
}