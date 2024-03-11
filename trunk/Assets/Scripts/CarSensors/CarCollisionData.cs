using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarCollisionData : MonoBehaviour
{
    public bool respawnCarOnHit = true;
    public bool isCollide = false;
    // init car
    private GameObject car;
    private Transform spawnPoint;

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

    void _processingCollision(Collision collision)
    {
        if (collision.gameObject.layer != LayerMask.NameToLayer("Ground"))
        {
            isCollide = true;
            if (respawnCarOnHit) { TeleportToSpawn(); }
        }
    }
    void OnCollisionEnter(Collision collision)
    {
        _processingCollision(collision);
    }
    void OnCollisionStay(Collision collision)
    {
        _processingCollision(collision);
    }
    void OnCollisionExit(Collision collision)
    {
        isCollide = false;
    }

}