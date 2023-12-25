using UnityEngine;

public class CameraData : MonoBehaviour
{
    public int cameraFixedSize = 64;
    private Camera targetCamera;
    private RenderTexture targetTexture;
    private Texture2D texture;
    private void Start()
    {
        // create camera and texture object
        targetCamera = GetComponent<Camera>();
        targetTexture = new RenderTexture(cameraFixedSize, cameraFixedSize, 24);
        targetTexture.filterMode = FilterMode.Point; // not compress image
        texture = new Texture2D(cameraFixedSize, cameraFixedSize, TextureFormat.RGB24, false);
    }
    public byte[] getFrameInBytes()
    {
        // Set target texture once
        targetCamera.targetTexture = targetTexture;
        targetCamera.Render();
        // Read pixels directly into the existing texture
        var currentRT = RenderTexture.active;
        RenderTexture.active = targetTexture;
        texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        texture.Apply();
        // Convert the Texture2D to ByteString
        byte[] imageBytes = texture.EncodeToPNG();
        // Reset camera settings
        RenderTexture.active = currentRT;
        targetCamera.targetTexture = null;
        // return frame
        return imageBytes;
    }

    private void OnDestroy()
    {
        Destroy(targetTexture);
    }
}