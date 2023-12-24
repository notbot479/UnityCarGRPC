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
    
    // grpc client and server
    private GrpcChannel channel;
    private Communication.CommunicationClient client;
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
        // init camera and distance sensors
        RaySensors = GameObject.Find("RaySensors");
        CarCamera = GameObject.Find("Camera");
    }

    public async void LateUpdate()
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
        Debug.Log(command);
    }
}