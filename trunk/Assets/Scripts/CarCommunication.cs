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
    public string serverDomain = "localhost";
    private string serverApiUrl;
    public int serverPort = 50051;
    public bool sendRequestToServer = true;
    public bool moveCarByAI = true;
   
    // grpc client & grpc channel
    private GrpcChannel channel;
    private Communication.CommunicationClient client;
    private bool serverConnectionError = false;
    private bool processingUpdate = true;
    private bool processingRespawn = false;
    // car & car data
    private GameObject car;
    private string carID;
    private bool carCollisionData;
    private string command;
    // camera & camera data
    private GameObject carCamera;
    private byte[] cameraImage;
    private bool boxesInCameraView; 
    // sensors & sensors data
    private GameObject carDistanceSensors;
    private Dictionary<string,float> distanceSensorsData;
    // router & router data
    private GameObject carRouterReceiver;
    private List<Tuple<string, float>> routersData;
    

    public void Start()
    {
        // create grpc client and channel
        serverApiUrl = $"http://{serverDomain}:{serverPort}";
        var channelOptions = new GrpcChannelOptions
        {
            HttpHandler = new YetAnotherHttpHandler { Http2Only = true },
            DisposeHttpClient = true
        };
        if (sendRequestToServer)
        {
            channel = GrpcChannel.ForAddress(serverApiUrl, channelOptions);
            client = new Communication.CommunicationClient(channel);
        }
        // init game objects
        car = GameObject.Find("Car");
        carCamera = GameObject.Find("Camera");
        carDistanceSensors = GameObject.Find("RaySensors");
        carRouterReceiver = GameObject.Find("CarRouterReceiver");
    }

    public void Update()
    {
        if (!processingUpdate) { return; }
        // get data: car state, camera image, sensors data, routers data
        carID = car.GetComponent<CarInfo>().ID;
        cameraImage = carCamera.GetComponent<CameraData>().getCameraImageInBytes();
        boxesInCameraView =  carCamera.GetComponent<CameraData>().getBoxesInCameraViewStatus();
        distanceSensorsData = carDistanceSensors.GetComponent<RaySensorsData>().GetSensorsData();
        routersData = carRouterReceiver.GetComponent<CarRouterReceiver>().GetRoutersData();
        carCollisionData = car.GetComponent<CarCollisionData>().isCollide;
        // skip send request to server
        if (!sendRequestToServer) { return; }
        // processing respawn car (skip send request)
        if (carCollisionData && processingRespawn) { return; }
        else if (processingRespawn) { processingRespawn = false; }
        
        // create grpc request
        var request = new ClientRequest
        {
            CarId = carID,
            CameraImage = ByteString.CopyFrom(cameraImage),
            DistanceSensorsData = new DistanceSensorsData
            {
                FrontLeftDistance = distanceSensorsData["FrontLeftDistance"],
                FrontDistance = distanceSensorsData["FrontDistance"],
                FrontRightDistance = distanceSensorsData["FrontRightDistance"],
                BackLeftDistance = distanceSensorsData["BackLeftDistance"],
                BackDistance = distanceSensorsData["BackDistance"],
                BackRightDistance = distanceSensorsData["BackRightDistance"],
            },
            CarCollisionData = carCollisionData,
            BoxesInCameraView = boxesInCameraView,
            QrCodeMetadata = "metadata", // TODO not implemented
        };
        foreach (var t in routersData)
        {
            request.RoutersData.Add(new RouterData { Id = t.Item1, Rssi = t.Item2 });
        }
        
        // send request using grpc, receive command
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
                Debug.Log("Failed connect to server: " + serverApiUrl);
                serverConnectionError = true;
            }
            return;
        }
        
        // processing command from server
        if (!moveCarByAI) { return; }
        try{
            if (command == "Respawn" && !processingRespawn)
            {
                processingRespawn = true;
                car.GetComponent<CarCollisionData>().TeleportToSpawn();
            }
            else if (command == "Poweroff")
            {
                Debug.Log("Poweroff"); // TODO end simulation
                processingUpdate = false;
            }
            else
            {
                car.GetComponent<CarControllerAdvanced>().CarMove(command);
            }
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