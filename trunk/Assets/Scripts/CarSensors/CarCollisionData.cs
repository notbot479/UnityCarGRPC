using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarCollisionData : MonoBehaviour
{
    public bool respawnCarOnHit = true;
    public bool isCollide = false;
    // init car
    public GameObject car;
    public Transform spawnPoint;

    public void Start()
    {
        car = GameObject.Find("Car");
        spawnPoint = GameObject.Find("SpawnPoint").transform;
    }
    public void TeleportToSpawn()
    {
        transform.forward = spawnPoint.forward;
        transform.rotation = spawnPoint.rotation;
        transform.position = spawnPoint.position;
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer != LayerMask.NameToLayer("Ground"))
        {
            isCollide = true;
            if (respawnCarOnHit) { TeleportToSpawn(); }
        }
    }
    void OnCollisionExit(Collision collision)
    {
        isCollide = false;
    }

}
