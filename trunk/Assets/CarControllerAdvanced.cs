using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class CarControllerAdvanced : MonoBehaviour {

    public WheelSet frontSet;
    public WheelSet backSet;

    public float steerMax; // Max angle wheels rotate (In degrees)
    public float motorMax; // Max torque of motors (In N*m)
    public float brakeMax; // Max braking (In N*m)

    public Vector3 objectCentreOfMass; // Allows the centre of mass to be offset (relative, in metres)

    public bool debugDisplay = true; // Whether or not debug information should be displayed

    float steer = 0.0f;
    float forward = 0.0f;
    float back = 0.0f;
    float speed = 0.0f; // The net velocity of the object (In m/s)
    float motor = 0.0f;
    float brake = 0.0f;

    bool reverse = false;
   
	// Use this for initialization
	void Start() {
        // Set centre of mass to what is defined in the inspector
        GetComponent<Rigidbody>().centerOfMass += objectCentreOfMass;

        // Setup the wheel sets
        frontSet.Init();
        backSet.Init();
	}

    void OnGUI() {
        // Debug log
        if (debugDisplay) {
            GUI.Label(new Rect(10.0f, 10.0f, 100.0f, 20.0f), "Speed: " + speed.ToString());
            GUI.Label(new Rect(10.0f, 30.0f, 100.0f, 20.0f), "Steer: " + steer.ToString());
            GUI.Label(new Rect(10.0f, 50.0f, 100.0f, 20.0f), "Motor: " + motor.ToString());
            GUI.Label(new Rect(10.0f, 70.0f, 100.0f, 20.0f), "Brake: " + brake.ToString());
            GUI.Label(new Rect(10.0f, 90.0f, 500.0f, 20.0f), "Left Motor Torque: " + backSet.GetTorque(Side.left));
            GUI.Label(new Rect(10.0f, 110.0f, 500.0f, 20.0f), "Right Motor Torque: " + backSet.GetTorque(Side.right));
        }
    }

    void Update() {
        // Retrieve Input
        steer = Mathf.Clamp(Input.GetAxis("Horizontal"), -1, 1);
        forward = Mathf.Clamp(Input.GetAxis("Vertical"), 0, 1);
        back = -1 * Mathf.Clamp(Input.GetAxis("Vertical"), -1, 0);

        frontSet.UpdateWheels();
        backSet.UpdateWheels();

        // Calculate the speed of the 
        speed = GetComponent<Rigidbody>().velocity.magnitude;

        if ((int)speed == 0) { // Cast as an (int) due to the accuracy of floating point and the physics setup, speed will never be exactly zero
            if (back > 0)
                reverse = true;
            if (forward > 0)
                reverse = false;
        }

        if (reverse) { 
            motor = -1 * back;
            brake = forward;
        }
        else {
            motor = forward;
            brake = back;
        }
    }

	// Update is called once per frame
	void FixedUpdate() {

        // Throttle
        frontSet.Throttle(Side.left, motor, motorMax);
        frontSet.Throttle(Side.right, motor, motorMax);
        backSet.Throttle(Side.left, motor, motorMax);
        backSet.Throttle(Side.right, motor, motorMax);
        
        // Brakes
        frontSet.Brake(Side.left, brake, brakeMax);
        frontSet.Brake(Side.right, brake, brakeMax);
        backSet.Brake(Side.left, brake, brakeMax);
        backSet.Brake(Side.right, brake, brakeMax);

        // Steering (This needs to be changed to be dependent on rear motors)
        frontSet.Steer(Side.left, steer, steerMax);
        frontSet.Steer(Side.right, steer, steerMax);
        backSet.Steer(Side.left, steer, steerMax);
        backSet.Steer(Side.right, steer, steerMax);
	}
}
