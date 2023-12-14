using Grpc.Net.Client;
using StreamVideoService;
using Google.Protobuf;

using var channel = GrpcChannel.ForAddress("http://localhost:50051");
var client = new Video.VideoClient(channel);

 var random = new System.Random();
 byte[] imageBytes = new byte[10];
 random.NextBytes(imageBytes);
 var byteString = ByteString.CopyFrom(imageBytes);
 
 // Send video frame to gRPC server
 var request = new VideoFrameRequest { Chunk = byteString };
 var responce = await client.UploadVideoFrameAsync(request);
Console.WriteLine(responce);
