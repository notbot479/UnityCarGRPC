using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Cysharp.Net.Http;
using Grpc.Net.Client;
using Grpc.Core;

using GreeterClientApp;


public class CarGrpc : MonoBehaviour
{
    private GrpcChannel channel;
    private Greeter.GreeterClient client;

    async void Start()
    {
        channel = GrpcChannel.ForAddress("http://localhost:5281", new GrpcChannelOptions
        {
            HttpHandler = new YetAnotherHttpHandler { Http2Only = true },
            DisposeHttpClient = true
        });

        client = new Greeter.GreeterClient(channel);

        var reply = await client.SayHelloAsync(new HelloRequest { Name = "World" });
        Debug.Log(reply.Message);
    }

    async void Update()
    {
        //await client.SayHelloAsync(new HelloRequest { Name = "Update" });
    }

    // Don't forget to clean up the channel when the MonoBehaviour is destroyed
    void OnDestroy()
    {
        channel?.ShutdownAsync().Wait();
    }
}

