using System.Collections.Generic;
using UnityEngine;

using Cysharp.Net.Http;
using Grpc.Net.Client;
using Grpc.Core;

using Google.Protobuf;
using CarCommunicationApp;


public class CarCommunication : MonoBehaviour
{
    private GrpcChannel channel;
    private Communication.CommunicationClient client;
   
    public GameObject RaySensors;
    public Dictionary<string,float> distanceSensorsData;

    public GameObject CarCamera;
    public byte[] videoFrame;

    void Start()
    {
        // create grpc channel and client
        channel = GrpcChannel.ForAddress("http://localhost:50051", new GrpcChannelOptions
        {
            HttpHandler = new YetAnotherHttpHandler { Http2Only = true },
            DisposeHttpClient = true
        });
        client = new Communication.CommunicationClient(channel);
    }

    async void Update()
    {
        // get data from sensors
        GameObject RaySensors = GameObject.Find("RaySensors");
        distanceSensorsData = RaySensors.GetComponent<DistanceSensorsData>().GetSensorsData();
        // get picture from camera
        GameObject CarCamera = GameObject.Find("Camera");
        videoFrame = CarCamera.GetComponent<CameraData>().getFrameInBytes();
        var videoFrameByteString = ByteString.CopyFrom(videoFrame);
        // send data using grpc
        var request = new ClientRequest
        {
            VideoFrame = videoFrameByteString,
            SensorsData = new SensorsData
            {
                FrontLeftDistance = 1.0f,
                FrontDistance = 2.0f,
                FrontRightDistance = 3.0f,
                BackLeftDistance = 4.0f,
                BackDistance = 5.0f,
                BackRightDistance = 6.0f
            },
        };
        var response = await client.SendRequestAsync(request);
        Debug.Log(response);
    }

}