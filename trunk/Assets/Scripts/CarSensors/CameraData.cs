using UnityEngine;
using ZXing;

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

    public void teleportCameraToBase()
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
    public bool getBoxesInCameraViewStatus()
    {
        OrderBox[] Boxes = FindObjectsOfType<OrderBox>();
        foreach (OrderBox Box in Boxes)
        {
            if (IsBoxVisible(Box)) { return true; }
        }
        return false;
    }
    private bool IsBoxVisible(OrderBox Box)
    {
        var bounds = Box.GetComponent<Collider>().bounds;
        var cameraFrustum = GeometryUtility.CalculateFrustumPlanes(targetCamera);
        // simple check box visibility (in view + no walls)
        bool boxInView = GeometryUtility.TestPlanesAABB(cameraFrustum, bounds);
        if (!boxInView) { return false; }
        bool boxInClearView = IsBoxClearView(Box);
        if (!boxInClearView) { return false; }
        return true;
    }
    private bool IsBoxClearView(OrderBox Box)
    {
        Transform cameraTransform = targetCamera.transform;
        Transform  boxTransform = Box.transform;
        Vector3 direction = boxTransform.position - cameraTransform.position;
        //Debug.DrawRay(cameraTransform.position, direction * 1000f, Color.red);
        RaycastHit hit;
        if (Physics.Raycast(cameraTransform.position, direction, out hit))
        {
            if (hit.transform != boxTransform) { return false; }
        }
        return true;
    }
    public byte[] getCameraImageInBytes()
    {
        // replace textures to custom target texture
        targetCamera.targetTexture = targetTexture;
        RenderTexture.active = targetTexture;
        targetCamera.Render();
        // read pixels from custom render texture
        texture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        byte[] imageBytes = texture.EncodeToPNG();
        // reset target texture and return bytes
        targetCamera.targetTexture = null;
        return imageBytes;
    }
    public string getQRCodeMetadata()
    {
        // replace textures to custom target texture
        targetCamera.targetTexture = targetTextureFullscreen;
        RenderTexture.active = targetTextureFullscreen;
        targetCamera.Render();
        // read pixels from custom render texture
        textureFullscreen.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        // grab data from qr code
        IBarcodeReader barcodeReader = new BarcodeReader();
        var result = barcodeReader.Decode(
            textureFullscreen.GetPixels32(), 
            textureFullscreen.width,
            textureFullscreen.height
        );
        // reset target texture and return qr metadata
        targetCamera.targetTexture = null;
        if (result != null) { return result.Text; }
        return "";
    }
    private void OnDestroy()
    {
        Destroy(targetTexture);
        Destroy(targetTextureFullscreen);
    }
}