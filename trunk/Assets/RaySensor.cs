using UnityEngine;
public class RaySensor : MonoBehaviour
{
    public string sensorName = "Sensor";
    public bool sensorShow = true;
    public Color rayColor = Color.blue;

    void Update()
    {
        // init ray
        var origin = transform.position;
        var direction = transform.right;        
        Ray ray = new Ray(origin, direction);
        // show rays in debug mode
        if (sensorShow) {Debug.DrawRay(origin, direction * 100f, rayColor);}
        // check collision
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit))
        {
            Debug.Log(sensorName+": "+hit.distance.ToString());
        }
    }

}

