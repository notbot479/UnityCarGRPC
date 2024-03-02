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
        # connect corner routers
        cls._graph.add_edge('1', '2', 2)
        cls._graph.add_edge('2', '3', 2)
        cls._graph.add_edge('3', '4', 2)
        cls._graph.add_edge('4', '5-1', 1)
        cls._graph.add_edge('5-1', '6', 3)
        cls._graph.add_edge('6', '7', 3)
        cls._graph.add_edge('7', '9', 2)
        # connect with middle router
        cls._graph.add_edge('2', '8', 1)
        cls._graph.add_edge('3', '8', 1)
        cls._graph.add_edge('4', '8', 2)
        cls._graph.add_edge('6', '8', 1)
        cls._graph.add_edge('7', '8', 1)
        cls._graph.add_edge('9', '8', 1)

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
