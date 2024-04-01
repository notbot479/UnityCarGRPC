using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;

public class CarControllerAdvanced : MonoBehaviour {
    private string UserMoveCommand = "_UserInput";

    public bool moveCarByUser = true;

    public WheelSet frontSet;
    public WheelSet backSet;

    public float steerMax; // Max angle wheels rotate (In degrees)
    public float motorMax; // Max torque of motors (In N*m)
    public float brakeMax; // Max braking (In N*m)

    public Vector3 objectCentreOfMass; // Allows the centre of mass to be offset (relative, in metres)
    public bool debugDisplay = true; // Whether or not debug information should be displayed
    public float speed = 0.0f; // The net velocity of the object (In m/s)
    
    float steer = 0.0f;
    float forward = 0.0f;
    float back = 0.0f;
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
            GUI.Label(new Rect(10.0f, 50.0f, 100.0f, 20.0f), "Motor: " + (-1 * motor).ToString());
            GUI.Label(new Rect(10.0f, 70.0f, 100.0f, 20.0f), "Brake: " + brake.ToString());
        }
    }

    void Update(){
        if (!moveCarByUser) { return; }
        string command = UserMoveCommand;
        CarMove(command);
    }

    public void CarMove(string command) {
        // Retrieve Input
        if (command == UserMoveCommand) { 
            steer = Mathf.Clamp(Input.GetAxis("Horizontal"), -1, 1);
            forward = Mathf.Clamp(Input.GetAxis("Vertical"), 1, 0);
            back = -1 * Mathf.Clamp(Input.GetAxis("Vertical"), 0, -1);
        } else if (command == "Forward") {
            steer = 0.0f;
            forward = -1.0f;
            back = 0.0f;
        } else if (command == "Backward") {
            steer = 0.0f;
            forward = 1.0f;
            back = 0.0f;
        } else if (command == "Left") {
            steer = -1.0f;
            forward = -1.0f;
            back = 0.0f;
        } else if (command == "Right") {
            steer = 1.0f;
            forward = -1.0f;
            back = 0.0f;
        } else if (command == "Stop") {
            steer = 0.0f;
            forward = 1f;
            back = 1f;
        }
        //Debug.Log($"{steer} {forward} {back}");
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
