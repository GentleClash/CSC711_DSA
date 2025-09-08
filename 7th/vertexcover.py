import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Set, List, Tuple, Dict, Optional, Iterable
from matplotlib.artist import Artist
from enum import Enum
import os

class ApproximationMethod(Enum):
    TWO_APPROXIMATION = "2-approximation"
    GREEDY = "greedy"

class VertexCover:
    def __init__(self, edges: Optional[List[Tuple[int, int]]] = None, 
                 vertices: Optional[List[int]] = None):
        self.graph = nx.Graph()
        self.original_graph = None
        self.vertex_cover = set()
        self.algorithm_stats = {}
        self.optimal_lower_bound = 0
        self.pos = None
        
        if edges:
            self.add_edges(edges)
        if vertices:
            self.graph.add_nodes_from(vertices)
    
    def add_edges(self, edges: List[Tuple[int, int]]) -> None:
        self.graph.add_edges_from(edges)
        self.original_graph = self.graph.copy()
        self.pos = nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
        self._calculate_lower_bound()
    
    def add_edge(self, u: int, v: int) -> None:
        self.graph.add_edge(u, v)
        if self.original_graph is None:
            self.original_graph = self.graph.copy()
        else:
            self.original_graph.add_edge(u, v)
        self.pos = nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
        self._calculate_lower_bound()
    
    def _calculate_lower_bound(self) -> None:
        if self.graph.edges():
            matching = nx.max_weight_matching(self.graph)
            self.optimal_lower_bound = len(matching)
        else:
            self.optimal_lower_bound = 0
    
    def two_approximation_vertex_cover(self, animate: bool = False, save_path: str = "two_approx.gif") -> Set[int]:
        if not self.graph.edges():
            return set()
        
        temp_graph = self.graph.copy()
        cover = set()
        iterations = 0
        edges_in_matching = []
        animation_frames = []
        
        if animate:
            animation_frames.append({
                'cover': set(),
                'current_edge': None,
                'title': 'Initial Graph',
                'remaining_edges': set(temp_graph.edges())
            })
        
        while temp_graph.edges():
            edge = next(iter(temp_graph.edges()))
            u, v = edge
            edges_in_matching.append(edge)
            
            if animate:
                animation_frames.append({
                    'cover': cover.copy(),
                    'current_edge': edge,
                    'title': f'Selected edge ({u}, {v})',
                    'remaining_edges': set(temp_graph.edges())
                })
            
            cover.add(u)
            cover.add(v)
            
            edges_to_remove = []
            for current_edge in temp_graph.edges():
                if u in current_edge or v in current_edge:
                    edges_to_remove.append(current_edge)
            
            temp_graph.remove_edges_from(edges_to_remove)
            iterations += 1
            
            if animate:
                animation_frames.append({
                    'cover': cover.copy(),
                    'current_edge': None,
                    'title': f'Added vertices {u}, {v} to cover',
                    'remaining_edges': set(temp_graph.edges())
                })
        
        if animate:
            animation_frames.append({
                'cover': cover.copy(),
                'current_edge': None,
                'title': f'Final Cover: {sorted(cover)} (Size: {len(cover)})',
                'remaining_edges': set()
            })
            self._create_animation(animation_frames, save_path, "2-Approximation Algorithm")
        
        approximation_ratio = len(cover) / max(self.optimal_lower_bound, 1)
        
        self.vertex_cover = cover
        self.algorithm_stats['2-approximation'] = {
            'cover_size': len(cover),
            'iterations': iterations,
            'matching_edges': len(edges_in_matching),
            'lower_bound': self.optimal_lower_bound,
            'approximation_ratio': approximation_ratio,
            'theoretical_ratio': '≤ 2',
            'actual_ratio': f'{approximation_ratio:.2f}'
        }
        
        return cover
    
    def greedy_vertex_cover(self, animate: bool = False, save_path: str = "greedy.gif") -> Set[int]:
        if not self.graph.edges():
            return set()
        
        temp_graph = self.graph.copy()
        cover = set()
        edges_covered_total = 0
        total_edges = len(self.graph.edges())
        iteration_details = []
        animation_frames = []
        
        if animate:
            animation_frames.append({
                'cover': set(),
                'highlighted_vertex': None,
                'title': 'Initial Graph',
                'degrees': dict(temp_graph.degree()) #type: ignore
            })
        
        while temp_graph.edges():
            max_degree = -1
            max_vertex = None
            
            for vertex in temp_graph.nodes():
                degree: int = temp_graph.degree[vertex] #type: ignore
                if degree > max_degree:
                    max_degree = degree
                    max_vertex = vertex
            
            if max_vertex is not None:
                edges_covered_this_round = temp_graph.degree[max_vertex] #type: ignore
                
                if animate:
                    animation_frames.append({
                        'cover': cover.copy(),
                        'highlighted_vertex': max_vertex,
                        'title': f'Max degree vertex: {max_vertex} (degree: {edges_covered_this_round})',
                        'degrees': dict(temp_graph.degree()) #type: ignore
                    })
                
                cover.add(max_vertex)
                edges_covered_total += edges_covered_this_round
                
                iteration_details.append({
                    'vertex': max_vertex,
                    'degree': edges_covered_this_round,
                    'cumulative_edges': edges_covered_total
                })
                
                neighbors = list(temp_graph.neighbors(max_vertex))
                for neighbor in neighbors:
                    temp_graph.remove_edge(max_vertex, neighbor)
                
                temp_graph.remove_node(max_vertex)
                
                if animate:
                    animation_frames.append({
                        'cover': cover.copy(),
                        'highlighted_vertex': None,
                        'title': f'Added vertex {max_vertex} to cover',
                        'degrees': dict(temp_graph.degree()) if temp_graph.nodes() else {} #type: ignore
                    })
        
        if animate:
            animation_frames.append({
                'cover': cover.copy(),
                'highlighted_vertex': None,
                'title': f'Final Cover: {sorted(cover)} (Size: {len(cover)})',
                'degrees': {}
            })
            self._create_animation(animation_frames, save_path, "Greedy Algorithm")
        
        approximation_ratio = len(cover) / max(self.optimal_lower_bound, 1)
        
        self.vertex_cover = cover
        self.algorithm_stats['greedy'] = {
            'cover_size': len(cover),
            'edges_covered': edges_covered_total,
            'total_edges': total_edges,
            'iterations': len(iteration_details),
            'lower_bound': self.optimal_lower_bound,
            'approximation_ratio': approximation_ratio,
            'theoretical_ratio': 'O(log V) worst case, often much better',
            'actual_ratio': f'{approximation_ratio:.2f}',
            'iteration_details': iteration_details
        }
        
        return cover
    
    def _create_animation(self, frames, save_path, algorithm_name) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate_frame(frame_idx) -> Iterable[Artist]: #type: ignore
            ax.clear()
            frame = frames[frame_idx]
            
            if 'remaining_edges' in frame and self.pos is not None:
                remaining_edges = frame['remaining_edges']
                if remaining_edges:
                    nx.draw_networkx_edges(self.graph, self.pos, 
                                         edgelist=remaining_edges,
                                         alpha=0.6, width=2, edge_color='gray', ax=ax)
                
                all_edges = set(self.graph.edges())
                covered_edges = all_edges - remaining_edges
                if covered_edges:
                    nx.draw_networkx_edges(self.graph, self.pos,
                                         edgelist=covered_edges,
                                         alpha=0.3, width=1, edge_color='lightgray',
                                         style='dashed', ax=ax)
            elif self.pos is not None:
                nx.draw_networkx_edges(self.graph, self.pos, alpha=0.6, width=2, 
                                     edge_color='gray', ax=ax)
            
            if 'current_edge' in frame and frame['current_edge'] and self.pos is not None:
                nx.draw_networkx_edges(self.graph, self.pos,
                                     edgelist=[frame['current_edge']],
                                     edge_color='orange', width=4, ax=ax)
            
            cover_vertices = list(frame['cover'])
            non_cover_vertices = [v for v in self.graph.nodes() if v not in frame['cover']]
            
            if cover_vertices and self.pos is not None:
                nx.draw_networkx_nodes(self.graph, self.pos, nodelist=cover_vertices,
                                     node_color='red', node_size=800, alpha=0.8, ax=ax)
            
            if non_cover_vertices and self.pos is not None:
                node_colors: List[str] = []
                node_sizes: List[int] = []
                for v in non_cover_vertices:
                    if 'highlighted_vertex' in frame and v == frame['highlighted_vertex']:
                        node_colors.append('orange')
                        node_sizes.append(900)
                    else:
                        node_colors.append('lightblue')
                        node_sizes.append(600)
                
                nx.draw_networkx_nodes(self.graph, self.pos, nodelist=non_cover_vertices,
                                     node_color=node_colors, node_size=node_sizes, #type: ignore
                                     alpha=0.8, ax=ax)
            
            nx.draw_networkx_labels(self.graph, self.pos, font_size=12,  #type: ignore
                                  font_weight='bold', ax=ax)
            
            if 'degrees' in frame and frame['degrees'] and self.pos is not None:
                for v in frame['degrees']:
                    if v in self.graph.nodes() and v in self.pos:
                        x, y = self.pos[v]
                        ax.text(x, y-0.15, f"d:{frame['degrees'][v]}", 
                               ha='center', va='top', fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.7))
            
            ax.set_title(f'{algorithm_name}\n{frame["title"]}', 
                        fontsize=12, fontweight='bold', pad=20)
            ax.axis('off')
        
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(frames),
                                     interval=2000, repeat=True, blit=False)
        
        writer = animation.PillowWriter(fps=1)
        anim.save(save_path, writer=writer)
        print(f"Animation saved as {save_path}")
        
        plt.show()
        plt.close()
    
    def find_vertex_cover(self, method: ApproximationMethod = ApproximationMethod.TWO_APPROXIMATION, 
                         animate: bool = False, save_path: Optional[str] = None) -> Set[int]:
        if save_path is None:
            save_path = f"{method.value}_animation.gif"
            
        if method == ApproximationMethod.GREEDY:
            return self.greedy_vertex_cover(animate=animate, save_path=save_path)
        elif method == ApproximationMethod.TWO_APPROXIMATION:
            return self.two_approximation_vertex_cover(animate=animate, save_path=save_path)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def verify_vertex_cover(self, cover: Optional[Set[int]] = None) -> bool:
        if cover is None:
            cover = self.vertex_cover
        
        for edge in self.graph.edges():
            u, v = edge
            if u not in cover and v not in cover:
                return False
        
        return True
    
    def get_stats(self) -> Dict:
        return self.algorithm_stats.copy()
    
    def compare_algorithms_animated(self, save_dir: str = "animations") -> Dict:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        results = {}
        
        print("Running 2-Approximation with animation...")
        results['2-approximation'] = {
            'cover': self.find_vertex_cover(ApproximationMethod.TWO_APPROXIMATION, 
                                          animate=True, 
                                          save_path=os.path.join(save_dir, "two_approximation.gif")),
            'size': len(self.vertex_cover),
            'valid': self.verify_vertex_cover(),
            'approximation_ratio': self.algorithm_stats['2-approximation']['approximation_ratio'],
        }
        
        print("\nRunning Greedy with animation...")
        results['greedy'] = {
            'cover': self.find_vertex_cover(ApproximationMethod.GREEDY, 
                                          animate=True, 
                                          save_path=os.path.join(save_dir, "greedy.gif")),
            'size': len(self.vertex_cover),
            'valid': self.verify_vertex_cover(),
            'approximation_ratio': self.algorithm_stats['greedy']['approximation_ratio'],
        }
        
        return results
    
    def compare_algorithms(self) -> Dict:
        results = {}
        
        results['2-approximation'] = {
            'cover': self.two_approximation_vertex_cover(),
            'size': len(self.vertex_cover),
            'valid': self.verify_vertex_cover(),
            'approximation_ratio': self.algorithm_stats['2-approximation']['approximation_ratio'],
            'theoretical_bound': '≤ 2'
        }
        
        results['greedy'] = {
            'cover': self.greedy_vertex_cover(),
            'size': len(self.vertex_cover),
            'valid': self.verify_vertex_cover(),
            'approximation_ratio': self.algorithm_stats['greedy']['approximation_ratio'],
            'theoretical_bound': 'O(log V)'
        }
        
        return results

    def visualize(self, method: Optional[ApproximationMethod] = None,
                figsize: Tuple[int, int] = (12, 8),
                save_path: Optional[str] = None,
                show_labels: bool = True,
                compare_all: bool = False) -> None:
        if not self.graph.nodes():
            print("No graph to visualize. Please add edges first.")
            return
        
        if compare_all:
            self._visualize_comparison(figsize, save_path, show_labels)
            return
        
        if method is not None:
            self.find_vertex_cover(method)
        
        plt.figure(figsize=figsize)
        
        pos = nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
        
        nx.draw_networkx_edges(self.graph, pos, alpha=0.6, width=2, edge_color='gray')
        
        cover_vertices = [v for v in self.graph.nodes() if v in self.vertex_cover]
        non_cover_vertices = [v for v in self.graph.nodes() if v not in self.vertex_cover]
        
        if cover_vertices:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=cover_vertices,
                                node_color='red', node_size=800, alpha=0.8,
                                label='Vertex Cover')
        
        if non_cover_vertices:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=non_cover_vertices,
                                node_color='lightblue', node_size=600, alpha=0.8,
                                label='Other Vertices')
        
        if show_labels:
            nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold')
        
        method_name = method.value if method else "Current"
        ratio_info = ""
        if method and method.value in self.algorithm_stats:
            stats = self.algorithm_stats[method.value]
            ratio_info = f"Approximation Ratio: {stats['actual_ratio']} (Bound: {stats['theoretical_ratio']})"
        
        plt.title(f'{method_name.title()} Vertex Cover\n'
                f'Cover Size: {len(self.vertex_cover)} | Lower Bound: {self.optimal_lower_bound}\n'
                f'{ratio_info}\n'
                f'Valid Cover: {self.verify_vertex_cover()}',
                fontsize=12, fontweight='bold')
        
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

    def _visualize_comparison(self, figsize: Tuple[int, int],
                            save_path: Optional[str], show_labels: bool) -> None:
        results = self.compare_algorithms()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Vertex Cover Algorithm Comparison with Approximation Ratios', 
                    fontsize=14, fontweight='bold')
        
        pos = nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
        
        ax = axes[0]
        plt.sca(ax)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.6, width=2, edge_color='gray')
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue',
                            node_size=600, alpha=0.8)
        if show_labels:
            nx.draw_networkx_labels(self.graph, pos, font_size=10)
        ax.set_title(f'Original Graph\n{len(self.graph.edges())} edges, {len(self.graph.nodes())} vertices\nLower Bound: {self.optimal_lower_bound}')
        ax.axis('off')
        
        algorithms = ['2-approximation', 'greedy']
        ax_indices = [1, 2]
        
        for alg, ax_idx in zip(algorithms, ax_indices):
            ax = axes[ax_idx]
            plt.sca(ax)
            
            cover = results[alg]['cover']
            cover_vertices = [v for v in self.graph.nodes() if v in cover]
            non_cover_vertices = [v for v in self.graph.nodes() if v not in cover]
            
            nx.draw_networkx_edges(self.graph, pos, alpha=0.6, width=2, edge_color='gray')
            
            if cover_vertices:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=cover_vertices,
                                    node_color='red', node_size=600, alpha=0.8)
            
            if non_cover_vertices:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=non_cover_vertices,
                                    node_color='lightblue', node_size=400, alpha=0.8)
            
            if show_labels:
                nx.draw_networkx_labels(self.graph, pos, font_size=8)
            
            ax.set_title(f'{alg.replace("-", " ").title()}\n'
                        f'Size: {results[alg]["size"]} | '
                        f'Ratio: {results[alg]["approximation_ratio"]:.2f}\n'
                        f'Bound: {results[alg]["theoretical_bound"]} | '
                        f'Valid: {results[alg]["valid"]}')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    print("=== Vertex Cover with Step-by-Step Animation ===")
    edges = [
        (1, 2), (1, 4), (2, 3), (2, 5),
        (3, 5), (3, 6), (3, 7), (5, 6)
    ]
    
    vc = VertexCover(edges)
    print("Graph edges:", edges)
    print("Graph nodes:", list(vc.graph.nodes()))
    print(f"Lower bound: {vc.optimal_lower_bound}")
    
    results = vc.compare_algorithms_animated()
    
    print(f"\n=== Animated Results ===")
    print(f"2-Approximation: Size={results['2-approximation']['size']}, "
          f"Ratio={results['2-approximation']['approximation_ratio']:.2f}")
    print(f"Greedy: Size={results['greedy']['size']}, "
          f"Ratio={results['greedy']['approximation_ratio']:.2f}")
    
    if results['greedy']['size'] < results['2-approximation']['size']:
        print("✓ Greedy performed better")
    elif results['greedy']['size'] > results['2-approximation']['size']:
        print("✓ 2-Approximation performed better")
    else:
        print("✓ Both algorithms found equal-sized covers")
    
    
    print("\nGenerating static comparison visualization...")
    vc.visualize(compare_all=True, figsize=(15, 5))