using UnityEngine;

public class CameraData : MonoBehaviour
{
    public int cameraFixedSize = 64;
    private Camera targetCamera;
    private GameObject cameraBase;
    private GameObject cameraBaseLookedAt;
    // init custom texture
    private RenderTexture targetTexture;
    private Texture2D texture;
    // init custom texture fullscreen
    private RenderTexture targetTextureFullscreen;
    private Texture2D textureFullscreen;

    private void teleportCameraToBase()
    {
        targetCamera.transform.position = cameraBase.transform.position;
        targetCamera.transform.LookAt(cameraBaseLookedAt.transform);
    }
    private void Start()
    {
        // grab camera and base objects
        cameraBase = GameObject.Find("CameraPlaceholderBase");
        cameraBaseLookedAt = GameObject.Find("CameraPlaceholderLookedAt");
        targetCamera = GetComponent<Camera>();
        // create custom render texture
        targetTexture = new RenderTexture(cameraFixedSize, cameraFixedSize, 16){
            filterMode = FilterMode.Point, // not compress image
        };
        texture = new Texture2D(cameraFixedSize, cameraFixedSize, TextureFormat.RGB24, false);
        // create fullscreen render texture
        targetTextureFullscreen = new RenderTexture(Screen.width, Screen.height, 16){
            filterMode = FilterMode.Point, // not compress image
        };
        textureFullscreen = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
    }
    public byte[] getCameraImageInBytes()
    {
        // replace textures to custom target texture
        targetCamera.targetTexture = targetTexture;
        RenderTexture.active = targetTexture;
        // teleport camera to base and render texture
        teleportCameraToBase();
        targetCamera.Render();
        // read pixels from custom render texture
        texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        byte[] imageBytes = texture.EncodeToPNG();
        // reset target texture and return bytes
        targetCamera.targetTexture = null;
        return imageBytes;
    }
    public byte[] getCameraImageFullScreenInBytes()
    {
        // replace textures to custom target texture
        targetCamera.targetTexture = targetTextureFullscreen;
        RenderTexture.active = targetTextureFullscreen;
        // teleport camera to base and render texture
        teleportCameraToBase();
        targetCamera.Render();
        // read pixels from custom render texture
        textureFullscreen.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        byte[] imageBytes = textureFullscreen.EncodeToPNG();
        // reset target texture and return bytes
        targetCamera.targetTexture = null;
        return imageBytes;
    }
    private void OnDestroy()
    {
        Destroy(targetTexture);
        Destroy(targetTextureFullscreen);
    }
}