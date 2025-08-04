from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import networkx as nx
from collections import defaultdict

@dataclass
class Graph:
    """Graph class to represent an undirected graph using an adjacency list."""
    vertices: int  
    edges: List[Tuple[int, int]]

@dataclass
class CliqueFinder:
    """Class to find all cliques in a graph using DFS-based enumeration."""
    graph: Graph

    def __post_init__(self) -> None:
        self.adjacency_list: Dict[int, Set[int]] = defaultdict(set)
        for edge in self.graph.edges:
            self.adjacency_list[edge[0]].add(edge[1])
            self.adjacency_list[edge[1]].add(edge[0])

        self.all_cliques: List[Set[int]] = []

    def is_clique(self, vertices: Set[int]) -> bool:
        """Check if a set of vertices forms a clique."""
        vertex_list: List[int] = list(vertices)
        for i in range(len(vertex_list)):
            for j in range(i + 1, len(vertex_list)):
                if vertex_list[j] not in self.adjacency_list[vertex_list[i]]:
                    return False
        return True

    def find_all_cliques_dfs(self) -> List[Set[int]]:
        """Find all maximal cliques using DFS."""
        self.all_cliques = []
        
        def dfs(r: Set[int], p: Set[int], x: Set[int]) -> None:
            """ DFS algorithm to find maximal cliques.
            Args:
                r: current clique being built
                p: potential vertices to add to the clique
                x: vertices already considered for the clique 
            
            Returns:
                None
            """
            if not p and not x:
                # Found a maximal clique
                if r:  # If not empty
                    self.all_cliques.append(r.copy())
                return
            
            # Choose pivot to minimize recursive calls
            pivot: int | None = next(iter(p.union(x))) if p.union(x) else None
            pivot_neighbors: Set[int] = self.adjacency_list[pivot] if pivot is not None else set()
            
            for v in p - pivot_neighbors:
                v_neighbors: Set[int] = self.adjacency_list[v]
                dfs(
                    r.union({v}),
                    p.intersection(v_neighbors),
                    x.intersection(v_neighbors)
                )
                p.remove(v)
                x.add(v)
        
        all_vertices: Set[int] = set(self.adjacency_list.keys())
        dfs(set(), all_vertices, set())

        # Remove any cliques with vertices <=2
        self.all_cliques = [clique for clique in self.all_cliques if len(clique) > 2]
        if not self.all_cliques:
            print("No cliques found.")
        return self.all_cliques

    def find_all_cliques_simple(self) -> List[Set[int]]:
        """Find all cliques (including non-maximal) using simple enumeration."""
        all_cliques: List[Set[int]] = []
        
        # Generate all possible subsets and check if they're cliques
        def generate_subsets(vertices: List[int], current_subset: Set[int], start_idx: int):
            # Check current subset if it's not empty
            if current_subset and self.is_clique(current_subset):
                all_cliques.append(current_subset.copy())
            
            # Generate larger subsets
            for i in range(start_idx, len(vertices)):
                current_subset.add(vertices[i])
                generate_subsets(vertices, current_subset, i + 1)
                current_subset.remove(vertices[i])
        
        vertices_list: List[int] = list(range(self.graph.vertices))
        generate_subsets(vertices_list, set(), 0)
        return all_cliques

    def count_all_cliques(self) -> int:
        """Count all cliques in the graph."""
        return len(self.find_all_cliques_simple())

    def count_maximal_cliques(self) -> int:
        """Count only maximal cliques in the graph."""
        return len(self.find_all_cliques_dfs())

    def print_cliques(self, only_maximal: bool = False) -> None:
        """Print all cliques found in the graph."""
        if only_maximal:
            cliques = self.find_all_cliques_dfs()
            print(f"Maximal cliques found: {len(cliques)}")
        else:
            cliques = self.find_all_cliques_simple()
            print(f"All cliques found: {len(cliques)}")

        if not cliques:
            print("No cliques found.")
            return

        
        for i, clique in enumerate(cliques, 1):
            print(f"Clique {i}: {sorted(list(clique))}")
    
    @staticmethod
    def visualize_graph(graph: Graph) -> None:
        """Visualize the graph using NetworkX and Matplotlib."""
        G = nx.Graph()
        nodes_in_edges = set()
        for u, v in graph.edges:
            G.add_edge(u, v)

        import matplotlib.pyplot as plt
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title("Graph Visualization")
        plt.savefig("graph_visualization.png")
        plt.show()

if __name__ == "__main__":
   
    graph = Graph(
        vertices=9,
        edges=[(1, 3), (1, 2),
               (2, 1), (2, 3),
               (3, 1), (3, 4),
               (4, 1), (4, 3), (4, 5), (4, 6),
               (5, 4), (5, 6), (5, 7), (5, 8),
               (6, 4), (6, 5), (6, 7), (6, 8),
               (7, 5), (7, 6), (7, 8),(7, 9),
               (8, 5), (8, 6), (8, 7),
               (9, 7)]
    )
        
    
    clique_finder2 = CliqueFinder(graph)
    clique_finder2.visualize_graph(graph)
    clique_finder2.print_cliques(only_maximal=True)
    print(f"Total cliques: {clique_finder2.count_all_cliques()}")
    print(f"Maximal cliques: {clique_finder2.count_maximal_cliques()}")