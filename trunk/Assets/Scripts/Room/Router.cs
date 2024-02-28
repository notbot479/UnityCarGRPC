using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using System;

public class Router : MonoBehaviour
{
    public int routerID = -1; // Unique router id, -1 -> not set
    public float maxDistance = 20f; // Maximum receive distance in meters
    private float minRSSI = -0f; // Minimum receive router strength
    private float maxRSSI = -100f; // Maximum receive router strength
    
    public bool sensorConnection = true;
    public readonly double mediumRSSIstart = -35f;
    public readonly double mediumRSSIend = -75f;
    public readonly Color rssiGoodColor = Color.green;
    public readonly Color rssiMediumColor = Color.yellow;
    public readonly Color rssiBadColor = Color.red;

    public double GetRSSI(Transform targetTransform)
    {
        var origin = transform.position;
        float distance = Vector3.Distance(origin, targetTransform.position);
        if (distance > maxDistance) { return float.NegativeInfinity; }
        // Simulate RSSI based on distance
        double rssi = CalculateRSSI(distance);
        // show rays in debug mode
        if (sensorConnection) 
        {
            Color rayColor = rssiGoodColor;
            if (rssi < mediumRSSIstart)
            {
                rayColor = rssiMediumColor;
            }
            if (rssi < mediumRSSIend)
            {
                rayColor = rssiBadColor;
            }
            Debug.DrawRay(origin, targetTransform.position - origin, rayColor); 
        }
        return rssi;
    }

    double CalculateRSSI(float distance)
    {
        double rssi = ((distance / maxDistance) * (maxRSSI - minRSSI)) + minRSSI;
        //double rssi = maxRSSI - 10 * Math.Log10(distance/maxDistance);
        return rssi;
    }
}
