using UnityEngine;
public class RaySensor : MonoBehaviour
{
    private readonly float maxDistanceRaycast = float.PositiveInfinity;
    private readonly float maxDistanceDebug = 1000f;
    public string sensorName = "Sensor";
    public Color rayColor = Color.blue;
    public bool sensorShow = true;
    public float GetDistanceToTarget()
    {
        // init ray
        var origin = transform.position;
        var direction = transform.right;
        Ray ray = new Ray(origin, direction);
        // show rays in debug mode
        if (sensorShow) { Debug.DrawRay(origin, direction * maxDistanceDebug, rayColor); }
        // check collision with all objects
        RaycastHit hit;
        float distanceToTarget = maxDistanceRaycast;
        if (Physics.Raycast(ray,out hit, maxDistanceRaycast)) { 
            distanceToTarget = hit.distance; 
        }
        //Debug.Log("["+sensorName+"] Distance: " + distanceToTarget.ToString());
        return distanceToTarget;
    }

}