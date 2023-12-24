using System.Collections.Generic;
using UnityEngine;

using Cysharp.Net.Http;
using Grpc.Net.Client;
using Grpc.Core;

using Google.Protobuf;
using CarCommunicationApp;


public class CarCommunication : MonoBehaviour
{
    public bool sendRequestToServer = true;
    public bool moveCarByAI = true;
    
    // grpc client and server
    private GrpcChannel channel;
    private Communication.CommunicationClient client;
    // car object
    private GameObject car;
    // sensors & sensors data
    private GameObject RaySensors;
    private Dictionary<string,float> raySensorsData;
    // camera & camera data
    private GameObject CarCamera;
    private byte[] videoFrame;
    private ByteString videoFrameByteString;

    public void Start()
    {
        // create grpc channel and client
        if (sendRequestToServer){
            channel = GrpcChannel.ForAddress("http://localhost:50051", new GrpcChannelOptions
            {
                HttpHandler = new YetAnotherHttpHandler { Http2Only = true },
                DisposeHttpClient = true
            });
            client = new Communication.CommunicationClient(channel);
        }
        // init car
        car =  GameObject.Find("Car");
        // init camera and distance sensors
        RaySensors = GameObject.Find("RaySensors");
        CarCamera = GameObject.Find("Camera");
    }

    public async void Update()
    {
        // get data from sensors and camera
        raySensorsData = RaySensors.GetComponent<RaySensorsData>().GetSensorsData();
        videoFrame = CarCamera.GetComponent<CameraData>().getFrameInBytes();
        videoFrameByteString = ByteString.CopyFrom(videoFrame);
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
        };
        // send data using grpc and receive command for car
        var response = await client.SendRequestAsync(request);
        string command = response.Command.Direction.ToString();
        // move car to some direction based on command from server
        if (!moveCarByAI) { return; }
        try{
            car.GetComponent<CarControllerAdvanced>().CarMove(command);
        } 
        catch{
            Debug.Log("Simulation is stopped. Ignore command from server");
        }
    }
}