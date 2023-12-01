using Grpc.Core;

class Program
{
    static async Task Main(string[] args)
    {
        Channel channel = new Channel("127.0.0.1:50051", ChannelCredentials.Insecure);
        var client = new Video.VideoClient(channel);

        using (var call = client.UploadVideo())
        {
            // Send video frames
            await SendVideoFrames(call.RequestStream);

            // Complete the gRPC call
            await call.RequestStream.CompleteAsync();

            var response = await call.ResponseAsync;
            Console.WriteLine($"Upload completed: {response.Success}");
        }

        channel.ShutdownAsync().Wait();
    }

    static async Task SendVideoFrames(IClientStreamWriter<VideoFrameRequest> requestStream)
    {
        // Read video frames and send individual frames
        using (var videoCapture = new VideoCapture("/home/max/Documents/Projects/UnityCarGRPC/test/client/1.mp4"))
        {
            Mat frame = new Mat();

            while (videoCapture.Read(frame))
            {
                using (var memoryStream = new MemoryStream())
                {
                    // Convert OpenCV Mat to byte array
                    frame.ToImage<Bgr, byte>().ToBitmap().Save(memoryStream, ImageFormat.Bmp);
                    var frameBytes = memoryStream.ToArray();

                    await requestStream.WriteAsync(new VideoFrameRequest { Frame = ByteString.CopyFrom(frameBytes) });
                }
            }
        }
    }
}

