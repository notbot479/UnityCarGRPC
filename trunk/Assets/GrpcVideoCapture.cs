using UnityEngine;

using Cysharp.Net.Http;
using Grpc.Net.Client;
using Grpc.Core;

using Google.Protobuf;
using StreamVideoService;

public class GrpcVideoCapture : MonoBehaviour
{
    private Camera targetCamera;
    private GrpcChannel channel;
    private Video.VideoClient client;

    void Start()
    {
        // create grpc channel and client
        channel = GrpcChannel.ForAddress("http://localhost:50051", new GrpcChannelOptions
        {
            HttpHandler = new YetAnotherHttpHandler { Http2Only = true },
            DisposeHttpClient = true
        });
        client = new Video.VideoClient(channel);
        // grab game object
        targetCamera = gameObject.GetComponent<Camera>();
    }
    async void Update()
    {
        RenderTexture renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
        targetCamera.targetTexture = renderTexture;
        
        // Create a new Texture2D and read the RenderTexture into it
        Texture2D texture = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
        targetCamera.Render();
        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        texture.Apply();
        
        // Convert the Texture2D to byteString
        byte[] imageBytes = texture.EncodeToJPG();
        var byteString = ByteString.CopyFrom(imageBytes);

        // Send video frame to gRPC server
        var request = new VideoFrameRequest { Chunk = byteString };
        var responce = await client.UploadVideoFrameAsync(request);
        Debug.Log(responce);

        // Reset camera settings
        targetCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(renderTexture);
    }

    void OnDestroy()
    {
        channel?.ShutdownAsync().Wait();
    }
}
