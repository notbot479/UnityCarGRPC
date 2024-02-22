from heapq import heappop, heappush


class WeightedGraph:
    def __init__(self) -> None:
        self.graph = {}

    def add_vertex(self, vertex:str) -> None:
        if vertex in self.graph: return
        self.graph[vertex] = {}

    def add_edge(self, from_vertex:str, to_vertex:str, weight:int) -> None:
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)
        self.graph[from_vertex][to_vertex] = weight
        self.graph[to_vertex][from_vertex] = weight

    def dijkstra_shortest_path(self, start_vertex:str, end_vertex:str) -> list[str]:
        # Initialize distances with infinity for all vertices
        distances = {vertex: float('inf') for vertex in self.graph}
        distances[start_vertex] = 0
        # Initialize predecessors to None for all vertices
        predecessors: dict[str,str | None] = {vertex: None for vertex in self.graph}
        # Priority queue to store vertices and their distances
        priority_queue = [(0, start_vertex)]

        while priority_queue:
            current_distance, current_vertex = heappop(priority_queue)
            # If current_vertex is the target vertex, construct and return the path
            if current_vertex == end_vertex:
                path = []
                while current_vertex is not None:
                    path.insert(0, current_vertex)
                    current_vertex = predecessors[current_vertex]
                return path
            # Skip this iteration if we've already found a better path to this vertex
            if current_distance > distances[current_vertex]:
                continue
            # Iterate through neighbors of current_vertex
            for neighbor, weight in self.graph[current_vertex].items():
                distance = current_distance + weight
                # If the new distance is shorter, update the distance and predecessor
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_vertex
                    heappush(priority_queue, (distance, neighbor))
        # If no path found, return an empty list
        return []


def _test():
    graph = WeightedGraph()

    graph.add_edge('1', '2', 2)
    graph.add_edge('2', '3', 2)
    graph.add_edge('3', '4', 2)
    graph.add_edge('4', '5', 1)
    graph.add_edge('5', '6', 2)
    graph.add_edge('6', '7', 3)
    graph.add_edge('7', '9', 2)

    graph.add_edge('2', '8', 1)
    graph.add_edge('3', '8', 1)
    graph.add_edge('4', '8', 2)
    graph.add_edge('6', '8', 1)
    graph.add_edge('7', '8', 1)
    graph.add_edge('9', '8', 1)

    shortest_distance = graph.dijkstra_shortest_path('9', '4')
    print(shortest_distance)

if __name__ == '__main__':
    _test()
