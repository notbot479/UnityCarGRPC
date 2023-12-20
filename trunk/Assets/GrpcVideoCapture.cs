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

    private RenderTexture renderTexture;
    private Texture2D texture;

    private void Start()
    {
        // create grpc channel and client
        channel = GrpcChannel.ForAddress("http://localhost:50051", new GrpcChannelOptions
        {
            HttpHandler = new YetAnotherHttpHandler { Http2Only = true },
            DisposeHttpClient = true
        });
        client = new Video.VideoClient(channel);
        // create camera and texture object
        targetCamera = GetComponent<Camera>();
        renderTexture = new RenderTexture(Screen.width, Screen.height, 24);
        texture = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
    }

    private void Update()
    {
        // Set target texture once
        targetCamera.targetTexture = renderTexture;
        targetCamera.Render();
        // Read pixels directly into the existing texture
        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        texture.Apply();
        // Convert the Texture2D to ByteString
        byte[] imageBytes = texture.EncodeToPNG();
        var byteString = ByteString.CopyFrom(imageBytes);
        // Send video frame to gRPC server
        var request = new VideoFrameRequest { Chunk = byteString };
        var response = client.UploadVideoFrame(request);
        // Reset camera settings
        targetCamera.targetTexture = null;
        RenderTexture.active = null;
    }
    private void OnDestroy()
    {
        channel?.ShutdownAsync().Wait();
        Destroy(renderTexture);
    }
}
