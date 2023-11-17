using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Cysharp.Net.Http;
using System.Net.Http;
using Grpc.Net.Client;
using Grpc.Core;

using GreeterServiceApp;

public class grpc : MonoBehaviour
{
    // Start is called before the first frame update
    async void Start()
    {
        // http test
        //using var handler = new YetAnotherHttpHandler();
        //var client = new HttpClient(handler);
        //var response = await client.GetStringAsync("https://http2.pro/");
        //Debug.Log(response);

        // grpc test
        using var handler = new YetAnotherHttpHandler();
        using var channel = GrpcChannel.ForAddress("http://localhost:50051/Greeter/SayHello", new GrpcChannelOptions() { HttpHandler = handler });
        Debug.Log(channel);
        var greeter = new Greeter.GreeterClient(channel);
        var result = await greeter.SayHelloAsync(new HelloRequest { Name = "Alice" });
        Debug.Log(result);
    }

    void Update()
    {
        
    }
}
