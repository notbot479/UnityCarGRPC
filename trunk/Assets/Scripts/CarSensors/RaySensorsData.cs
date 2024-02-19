using System.Collections.Generic;
using System.Collections;
using UnityEngine;

public class RaySensorsData : MonoBehaviour
{
    private GameObject SensorFrontLeft;
    private GameObject SensorFront;
    private GameObject SensorFrontRight;
    private GameObject SensorBackLeft;
    private GameObject SensorBack;
    private GameObject SensorBackRight;

    void Start()
    {
        SensorFrontLeft = GameObject.Find("SensorFrontLeft");
        SensorFront = GameObject.Find("SensorFront");
        SensorFrontRight = GameObject.Find("SensorFrontRight");
        SensorBackLeft = GameObject.Find("SensorBackLeft");
        SensorBack = GameObject.Find("SensorBack");
        SensorBackRight = GameObject.Find("SensorBackRight");
    }

    public Dictionary<string, float> GetSensorsData()
    {
        // get distance from sensors
        float front_left_distance = SensorFrontLeft.GetComponent<RaySensor>().GetDistanceToTarget();
        float front_distance = SensorFront.GetComponent<RaySensor>().GetDistanceToTarget();
        float front_right_distance = SensorFrontRight.GetComponent<RaySensor>().GetDistanceToTarget();
        float back_left_distance = SensorBackLeft.GetComponent<RaySensor>().GetDistanceToTarget();
        float back_distance = SensorBack.GetComponent<RaySensor>().GetDistanceToTarget();
        float back_right_distance = SensorBackRight.GetComponent<RaySensor>().GetDistanceToTarget();
        // convert to dict
        Dictionary<string, float> SensorsDataDict = new Dictionary<string, float>();
        SensorsDataDict.Add("FrontLeftDistance", front_left_distance);
        SensorsDataDict.Add("FrontDistance", front_distance);
        SensorsDataDict.Add("FrontRightDistance", front_right_distance);
        SensorsDataDict.Add("BackLeftDistance", back_left_distance);
        SensorsDataDict.Add("BackDistance", back_distance);
        SensorsDataDict.Add("BackRightDistance", back_right_distance);
        return SensorsDataDict;
    }
}
