using System.Collections.Generic;
using UnityEngine;

using Cysharp.Net.Http;
using Grpc.Net.Client;
using Grpc.Core;

using Google.Protobuf;
using CarCommunicationApp;
using System.Linq;
using System;
using Unity.VisualScripting;

public class CarCommunication : MonoBehaviour
{
    public bool sendRequestToServer = true;
    public bool moveCarByAI = true;
    public string serverApiUrl = "http://localhost:50051";
    private bool serverConnectionError = false;

    // grpc client and server
    private GrpcChannel channel;
    private Communication.CommunicationClient client;
    // car & car data
    private GameObject car;
    private GameObject carRouter;
    private bool carCollideObstacle;
    private string command;
    // sensors & sensors data
    private GameObject RaySensors;
    private Dictionary<string,float> raySensorsData;
    // camera & camera data
    private GameObject CarCamera;
    private byte[] videoFrame;
    private ByteString videoFrameByteString;
    // router & router data
    private GameObject[] routers;
    private int routerID;
    private double routerRSSI;

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
        // init car
        car =  GameObject.Find("Car");
        carRouter = GameObject.Find("CarRouter");
        // init camera and distance sensors
        RaySensors = GameObject.Find("RaySensors");
        CarCamera = GameObject.Find("Camera");
        routers = GameObject.FindObjectsOfType<Router>().Select(x => x.gameObject).ToArray();
    }

    public void Update()
    {
        // get data from sensors and camera
        raySensorsData = RaySensors.GetComponent<RaySensorsData>().GetSensorsData();
        videoFrame = CarCamera.GetComponent<CameraData>().getFrameInBytes();
        videoFrameByteString = ByteString.CopyFrom(videoFrame);
        carCollideObstacle = car.GetComponent<CarCollisionData>().isCollide;
        //get data from routers
        foreach (GameObject router in routers)
        {
            var r = router.GetComponent<Router>();
            routerRSSI =  r.GetRSSI(carRouter.transform);
            routerID = r.routerID;
            if (routerRSSI != float.NegativeInfinity) {
                Debug.Log($"Router ID: {routerID}, RSSI: {routerRSSI}");
            }   
        }
        // create grpc request
        if (!sendRequestToServer) { return; }
        var request = new ClientRequest
        {
            VideoFrame = videoFrameByteString,
            SensorsData = new SensorsData
            {
                FrontLeftDistance = raySensorsData["FrontLeftDistance"],
                FrontDistance = raySensorsData["FrontDistance"],
                FrontRightDistance = raySensorsData["FrontRightDistance"],
                BackLeftDistance = raySensorsData["BackLeftDistance"],
                BackDistance = raySensorsData["BackDistance"],
                BackRightDistance = raySensorsData["BackRightDistance"],
            },
            CarCollideObstacle = carCollideObstacle,
        };
        // send data using grpc and receive command for car
        try
        {
            var response = client.SendRequest(request);
            command = response.Command.Direction.ToString();
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
        // move car to some direction based on command from server
        if (!moveCarByAI) { return; }
        try{
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