using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using System;

public class Router : MonoBehaviour
{
    public int routerID = -1; // Unique router id, -1 -> not set
    public float maxDistance = 30f; // Maximum receive distance in meters
    public float maxRSSI = -100f; // Maximum receive router strength
    public float minRSSI = -0f; // Minimum receive router strength

    public bool sensorConnection = true;
    public readonly Color rayColor = Color.red;

    public double GetRSSI(Transform targetTransform)
    {
        var origin = transform.position;
        float distance = Vector3.Distance(origin, targetTransform.position);
        if (distance < maxDistance)
        {
            // show rays in debug mode
            if (sensorConnection) { Debug.DrawRay(origin, targetTransform.position - origin, rayColor); }
            // Simulate RSSI based on distance
            double rssi = CalculateRSSI(distance);
            return rssi;
        }
        else
        {
            return float.NegativeInfinity;
        }
    }

    double CalculateRSSI(float distance)
    {
        double rssi = ((distance / maxDistance) * (maxRSSI - minRSSI)) + minRSSI;
        //double rssi = maxRSSI - 10 * Math.Log10(distance/maxDistance);
        return rssi;
    }
}
