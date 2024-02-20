using System.Collections.Generic;
using System.Linq;
using UnityEngine;

using ZXing;

public class CameraData : MonoBehaviour
{
    private int cameraFixedSize = 64;
    private int cameraFullscreenFixedSize = 500;
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
        targetCamera = GetComponent<Camera>();
        cameraBase = GameObject.Find("CameraPlaceholderBase");
        cameraBaseLookedAt = GameObject.Find("CameraPlaceholderLookedAt");
        // create custom render texture
        targetTexture = new RenderTexture(cameraFixedSize, cameraFixedSize, 16){
            filterMode = FilterMode.Point, // not compress image
        };
        texture = new Texture2D(
            cameraFixedSize, 
            cameraFixedSize, 
            TextureFormat.RGB24, 
            false
        );
        // create fullscreen render texture
        targetTextureFullscreen = new RenderTexture(
            cameraFullscreenFixedSize, 
            cameraFullscreenFixedSize, 
            16
        );
        targetTextureFullscreen.filterMode = FilterMode.Point; // not compress image
        textureFullscreen = new Texture2D(
            cameraFullscreenFixedSize, 
            cameraFullscreenFixedSize, 
            TextureFormat.RGB24,
            false
        );
    }
    public float getVisibleDistanceToBox(OrderBox Box)
    {
        Transform cameraTransform = targetCamera.transform;
        Transform  boxTransform = Box.transform;
        Vector3 direction = boxTransform.position - cameraTransform.position;
        //Debug.DrawRay(cameraTransform.position, direction * 1000f, Color.red);
        RaycastHit hit;
        if (Physics.Raycast(cameraTransform.position, direction, out hit))
        {
            if (hit.transform == boxTransform) { return hit.distance; }
        }
        return float.PositiveInfinity;
    }
    public float getDistanceIfBoxInCameraView(OrderBox Box)
    {
        var bounds = Box.GetComponent<Collider>().bounds;
        var cameraFrustum = GeometryUtility.CalculateFrustumPlanes(targetCamera);
        // simple check box visibility (in view + no walls)
        bool boxInView = GeometryUtility.TestPlanesAABB(cameraFrustum, bounds);
        if (!boxInView) { return float.PositiveInfinity; }
        float distance = getVisibleDistanceToBox(Box);
        return distance;
    }

    public float getDistanceToNearestVisibleBox(){
        OrderBox[] Boxes = FindObjectsOfType<OrderBox>();
        // get distance for each box which visible
        List<float> distances = new List<float>();
        foreach (OrderBox Box in Boxes)
        {
            float distance = getDistanceIfBoxInCameraView(Box);
            distances.Add(distance);
        }
        float minDistance = distances.Min();
        return minDistance;
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