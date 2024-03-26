using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using System;

public class Router : MonoBehaviour
{
    public string routerID = "noID"; // Unique router id
    public float maxDistance = 20f; // Maximum receive distance in meters
    private float minRSSI = -0f; // Minimum receive router strength
    private float maxRSSI = -100f; // Maximum receive router strength
    
    public bool sensorConnection = true;
    public readonly double mediumRSSIstart = -10f;
    public readonly double mediumRSSIend = -50f;
    public readonly double badRSSIstart = -90f;
    public readonly Color rssiGoodColor = Color.green;
    public readonly Color rssiMediumColor = Color.yellow;
    public readonly Color rssiBadColor = Color.red;
    public readonly Color rssiVeryBadColor = Color.gray;

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
        if (rssi <= mediumRSSIstart && rssi >= mediumRSSIend)
        {
            rayColor = rssiMediumColor;
        }
        else if (rssi < mediumRSSIend && rssi >= badRSSIstart)
        {
            rayColor = rssiBadColor;
        }
        else if (rssi < badRSSIstart)
        {
            rayColor = rssiVeryBadColor;
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
