from .graph_manager import WeightedGraph


class RouteManager:
    _graph: WeightedGraph = WeightedGraph()

    @classmethod
    def get_shortest_route(
        cls,
        nearest_router_id: str, 
        target_router_id: str,
    ) -> list[str]:
        cls.update_routes()
        route = cls._graph.dijkstra_shortest_path(
            start_vertex=nearest_router_id,
            end_vertex=target_router_id,
        )
        return route
    
    @classmethod
    def update_routes(cls) -> None:
        cls._graph = WeightedGraph()

        # router 1
        cls._graph.add_edge('1', '8-1', 2)
        cls._graph.add_edge('1', '8-2', 3)
        # router 2
        cls._graph.add_edge('2', '8-2', 1)
        # router 3
        cls._graph.add_edge('3', '8-2', 1)
        # router 4
        cls._graph.add_edge('4', '8-2', 1)
        cls._graph.add_edge('4', '5-1', 1)
        cls._graph.add_edge('4', '6-2', 3)
        # router 5-1
        cls._graph.add_edge('5-1', '5-2', 1)
        cls._graph.add_edge('5-1', '4', 1)
        cls._graph.add_edge('5-1', '8-2', 2)
        # router 5-2
        cls._graph.add_edge('5-2', '5-1', 1)
        # router 6-1
        cls._graph.add_edge('6-1', '6-2', 1)
        cls._graph.add_edge('6-1', '8-2', 1)
        cls._graph.add_edge('6-1', '4', 2)
        cls._graph.add_edge('6-1', '7', 3)
        # router 6-2
        cls._graph.add_edge('6-2', '6-1', 1)
        cls._graph.add_edge('6-2', '7', 3)
        cls._graph.add_edge('6-2', '8-1', 4)
        # router 7
        cls._graph.add_edge('7', '6-2', 3)
        cls._graph.add_edge('7', '9', 3)
        cls._graph.add_edge('7', '8-1', 3)
        cls._graph.add_edge('7', '6-1', 3)
        # router 8-1
        cls._graph.add_edge('8-1', '8-2', 1)
        cls._graph.add_edge('8-1', '1', 2)
        cls._graph.add_edge('8-1', '9', 2)
        cls._graph.add_edge('8-1', '7', 3)
        cls._graph.add_edge('8-1', '6-2', 4)
        # router 8-2
        cls._graph.add_edge('8-2', '8-1', 1)
        cls._graph.add_edge('8-2', '2', 1)
        cls._graph.add_edge('8-2', '3', 1)
        cls._graph.add_edge('8-2', '6-1', 1)
        cls._graph.add_edge('8-2', '4', 1)
        cls._graph.add_edge('8-2', '5-1', 3)
        cls._graph.add_edge('8-2', '1', 3)
        # router 9
        cls._graph.add_edge('9', '8-1', 2)
        cls._graph.add_edge('9', '7', 3)

def _test():
    nearest_router_id = '9'
    target_router_id = '5-1' 
    route = RouteManager.get_shortest_route(
        nearest_router_id=nearest_router_id,
        target_router_id=target_router_id,
    )
    print(route)

if __name__ == '__main__':
    _test()
