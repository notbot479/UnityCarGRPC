using UnityEngine;
using System.Collections;

public enum Side {
    left,
    right
};

[System.Serializable]
public class WheelSet {

    Quaternion angleCorrection = Quaternion.Euler(0.0f, 0.0f, 90.0f); // Create a Quaternion that is rotated 90 degrees in the z direction
    const int dps = 6; // Conversion from rpm to degrees per second (was -6)

    public GameObject wheelLeftGo;
    public Transform wheelLeftT;
    WheelCollider wheelLeftWc;

    public GameObject wheelRightGo;
    public Transform wheelRightT;
    WheelCollider wheelRightWc;

    public bool drive; // Is this wheel attached to a drive?
    public bool brakes; // Does it have brakes?
    public bool steering; // Does this wheel apply steer angle?

    float steerAngle;
    Vector3 wheelRightRotationCurrent;
    Vector3 wheelLeftRotationCurrent;
    Vector3 tempRotation = new Vector3(0.0f, 0.0f, 0.0f);

    /// <summary>
    /// Makes a wheel turn (Called in FixedUpdate)
    /// </summary>
    /// <param name="side">Which side of the wheel set should turn</param>
    /// <param name="motor">How fast the wheel should turn</param>
    /// <param name="max">Max speed of the wheel</param>
    public void Throttle(Side side, float motor, float max) {
        if (drive) {
            if (side == Side.left)
                wheelLeftWc.motorTorque = -1 * max * motor;
            else
                wheelRightWc.motorTorque = -1 * max * motor;
        }
    }

    /// <summary>
    /// Makes a wheel stop (Called in FixedUpdate)
    /// </summary>
    /// <param name="side">Which side of the wheel set should stop</param>
    /// <param name="motor">How fast the wheel should stop</param>
    /// <param name="max">Max amount of brakes</param>
    public void Brake(Side side, float brake, float max) {
        if (brakes) {
            if (side == Side.left)
                wheelLeftWc.brakeTorque = max * brake;
            else
                wheelRightWc.brakeTorque = max * brake;
        }
    }

    public void Steer(Side side, float amount, float max) {
        steerAngle = amount * max;

        if (steering) {
            if (side == Side.left)
                wheelLeftWc.steerAngle = steerAngle;
            else
                wheelRightWc.steerAngle = steerAngle;
        }
    }

    /// <summary>
    /// Store the handles for the wheel components
    /// </summary>
    public void Init() {
        wheelLeftWc = wheelLeftT.GetComponent<WheelCollider>();
        wheelRightWc = wheelRightT.GetComponent<WheelCollider>();

        wheelLeftRotationCurrent = wheelLeftGo.transform.eulerAngles;
        wheelRightRotationCurrent = wheelLeftGo.transform.eulerAngles;
    }

    /// <summary>
    /// Behind the scenes magic (Called in Update)
    /// </summary>
    public void UpdateWheels() {
        // Update location of rendered model (dependent on collision meshes)
        wheelLeftGo.transform.position = wheelLeftWc.transform.position;
        wheelRightGo.transform.position = wheelRightWc.transform.position;

        // Update the (y component) rotation of the wheels
        if (steering) {
            tempRotation.y = Mathf.Lerp(tempRotation.y, steerAngle, 0.2f);
            wheelRightRotationCurrent.y = tempRotation.y;
            wheelLeftRotationCurrent.y = tempRotation.y;
            wheelLeftGo.transform.localEulerAngles = wheelLeftRotationCurrent;
            wheelRightGo.transform.localEulerAngles = wheelRightRotationCurrent;
        }
        else {
            wheelLeftGo.transform.rotation = wheelLeftWc.transform.rotation * angleCorrection;
            wheelRightGo.transform.rotation = wheelRightWc.transform.rotation * angleCorrection;
        }

        // Update the (z component) rotation of the wheels (relative to current values)
        wheelLeftGo.transform.Rotate(0.0f, wheelLeftWc.rpm * dps * Time.deltaTime, 0.0f);
        wheelRightGo.transform.Rotate(0.0f, wheelRightWc.rpm * dps * Time.deltaTime, 0.0f);
    }

    /// <summary>
    /// Get the amount of Torque of a wheel
    /// </summary>
    /// <param name="side">Which side of the wheel you want the torque of</param>
    /// <returns>The ammount of Torque the wheel is supplying</returns>
    public string GetTorque(Side side) {
        if (side == Side.left)
            return wheelLeftWc.motorTorque.ToString();
        else
            return wheelRightWc.motorTorque.ToString();
    }
}