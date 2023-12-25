using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarCollisionData : MonoBehaviour
{
    public bool isCollide = false;

    void OnCollisionEnter(Collision collision)
    {
        isCollide = true;
    }
    void OnCollisionExit(Collision collision)
    {
        isCollide = false;
    }

}
