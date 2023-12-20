using UnityEngine;

public class RaySensor : MonoBehaviour
{
    public float sensorLength = 10f;
    public float sensorSpread = 30f;
    public Color rayColor = Color.red;

    void Update()
    {
        // Cast rays on the front sensors
        CastSensorRays(transform.forward, Vector3.forward);

        // Cast rays on the back sensors
        CastSensorRays(-transform.forward, Vector3.back);
    }

    void CastSensorRays(Vector3 direction, Vector3 sensorDirection)
    {
        for (int i = -1; i <= 1; i++)
        {
            // Calculate the angle for the current sensor
            float angle = i * sensorSpread;

            // Rotate the sensor direction based on the angle
            Vector3 rotatedDirection = Quaternion.Euler(0, angle, 0) * sensorDirection;

            // Cast a ray in the rotated direction
            Ray ray = new Ray(transform.position, direction + rotatedDirection);
            RaycastHit hit;

            // Check for collisions
            if (Physics.Raycast(ray, out hit, sensorLength))
            {
                Debug.DrawLine(ray.origin, hit.point, rayColor);
                
                // You can access the distance to the object with hit.distance
                float distanceToObstacle = hit.distance;

                // Do something with the distance information (e.g., apply brakes based on proximity)
            }
            else
            {
                // Draw the full sensor length if no collision
                Debug.DrawRay(ray.origin, ray.direction * sensorLength, rayColor);
            }
        }
    }
}

