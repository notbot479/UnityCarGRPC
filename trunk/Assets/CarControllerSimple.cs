using UnityEngine;
using System.Collections;

/*
 * Originally written in JS
 * http://carpe.com.au/slawia/2009/08/unity-wheel-collider-part-2/
*/

public class CarControllerSimple : MonoBehaviour {

    public WheelCollider wheelFrontLeft;
    public WheelCollider wheelFrontRight;
    public WheelCollider wheelBackLeft;
    public WheelCollider wheelBackRight;

    public float steerMax;
    public float motorMax;
    public float brakeMax;

    float steer = 0.0f;
    float motor = 0.0f;
    float brake = 0.0f;
    
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void FixedUpdate() {
        steer = Mathf.Clamp(Input.GetAxis("Horizontal"), -1, 1);
        motor = Mathf.Clamp(Input.GetAxis("Vertical"), 0, 1);
        brake = -1 * Mathf.Clamp(Input.GetAxis("Vertical"), -1, 0);

        wheelBackLeft.motorTorque = -1 * motorMax * motor;
        wheelBackRight.motorTorque = -1 * motorMax * motor;
        wheelBackLeft.brakeTorque = brakeMax * brake;
        wheelBackRight.brakeTorque = brakeMax * brake;

        wheelFrontLeft.steerAngle = steerMax * steer;
        wheelFrontRight.steerAngle = steerMax * steer;

	}
}
