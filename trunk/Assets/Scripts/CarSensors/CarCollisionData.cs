using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarCollisionData : MonoBehaviour
{
    public bool respawnCarOnHit = false;
    public bool isCollide = false;
    
    public bool respawnRandomize = true;
    public float respawnRadius = 1f; 
    public float respawnRotation = 10f; 

    private float outOfBoundsY = -3f;
    private GameObject car;
    private Transform spawnPoint;

    public void Start()
    {
        car = GameObject.Find("Car");
        spawnPoint = GameObject.Find("SpawnPoint").transform;
    }

    private void RandomizeCarState()
    {   
        // randomize car position based on current coords
        Vector2 randomDirection = Random.insideUnitCircle.normalized * respawnRadius;
        Vector3 newPosition = new Vector3(
            transform.position.x + randomDirection.x, 
            transform.position.y, 
            transform.position.z + randomDirection.y
        );
        transform.position = newPosition;
        // randomize car rotation based on current rotation 
        float delta = Random.Range(-respawnRotation, respawnRotation);
        Vector3 currentRotation = transform.rotation.eulerAngles;
        Vector3 targetRotation = currentRotation + new Vector3(0f, delta, 0f);
        transform.rotation = Quaternion.Euler(targetRotation);
    }

    public void TeleportToSpawn(bool randomizeSpawn = false)
    {
        // teleport car to spawn
        transform.forward = spawnPoint.forward;
        transform.rotation = spawnPoint.rotation;
        transform.position = spawnPoint.position;
        // randomize spawn
        if (randomizeSpawn) { RandomizeCarState(); }
    }

    void _processingCollision(Collision collision)
    {
        if (collision.gameObject.layer != LayerMask.NameToLayer("Ground"))
        {
            isCollide = true;
            if (respawnCarOnHit) { TeleportToSpawn(respawnRandomize); }
        }
    }

    void Update()
    {
        float carY = car.transform.position.y; 
        if (carY < outOfBoundsY)
        {
            TeleportToSpawn();
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
