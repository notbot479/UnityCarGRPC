using System.Collections.Generic;
using System;
using UnityEngine;

using System.Linq;

public class CarRouterReceiver : MonoBehaviour
{
    private GameObject carRouterReceiver;
    private GameObject[] routers;
    private double routerRSSI;
    private int routerID;

    void Start()
    {
        routers = GameObject.FindObjectsOfType<Router>().Select(x => x.gameObject).ToArray();
    }

    public List<Tuple<string, double>> GetRoutersData()
    {
        List<Tuple<string, double>> routersDataList = new List<Tuple<string, double>>();
        foreach (GameObject router in routers)
        {
            var r = router.GetComponent<Router>();
            routerID = r.routerID;
            routerRSSI = r.GetRSSI(transform);
            if (routerRSSI != float.NegativeInfinity)
            {
                //Debug.Log($"Router ID: {routerID}, RSSI: {routerRSSI}");
                routersDataList.Add(Tuple.Create($"{routerID}",routerRSSI));
            }
        }
        return routersDataList;
    }
}
