using System.Collections.Generic;
using System;
using UnityEngine;

using System.Linq;

public class CarRouterReceiver : MonoBehaviour
{
    private GameObject carRouterReceiver;
    private GameObject[] routers;
    private string routerID;
    private float routerRSSI;

    void Start()
    {
        routers = GameObject.FindObjectsOfType<Router>().Select(x => x.gameObject).ToArray();
    }

    public List<Tuple<string, float>> GetRoutersData()
    {
        List<Tuple<string, float>> routersDataList = new List<Tuple<string, float>>();
        foreach (GameObject router in routers)
        {
            var r = router.GetComponent<Router>();
            routerID = r.routerID;
            routerRSSI = (float)r.GetRSSI(transform);
            if (routerRSSI != float.NegativeInfinity)
            {
                //Debug.Log($"Router ID: {routerID}, RSSI: {routerRSSI}");
                routersDataList.Add(Tuple.Create($"{routerID}",routerRSSI));
            }
        }
        return routersDataList;
    }
}
