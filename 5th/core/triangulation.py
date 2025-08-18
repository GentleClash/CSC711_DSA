

from typing import Dict, List, Tuple
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon


from core.monotonicitytester import Point, Polygon


class ChainType(Enum):
    """Enum to identify which chain a vertex belongs to"""
    LEFT = "left"
    RIGHT = "right"


class Triangle:
    """Represents a triangle formed during triangulation"""
    
    def __init__(self, p1: 'Point', p2: 'Point', p3: 'Point'):
        """
        Initialize triangle with three points
        
        Args:
            p1, p2, p3: Three Point objects forming the triangle
        """
        self.vertices = [p1, p2, p3]
        self.p1, self.p2, self.p3 = p1, p2, p3
    
    def area(self) -> float:
        """Calculate the area of the triangle using the cross product formula"""
        # Using the shoelace formula for triangle area
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = self.p3.x, self.p3.y
        
        return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)
    
    def centroid(self) -> 'Point':
        """Calculate the centroid of the triangle"""
        from __main__ import Point
        center_x = (self.p1.x + self.p2.x + self.p3.x) / 3
        center_y = (self.p1.y + self.p2.y + self.p3.y) / 3
        return Point(center_x, center_y)
    
    def __str__(self) -> str:
        return f"Triangle({self.p1}, {self.p2}, {self.p3})"
    
    def __repr__(self) -> str:
        return self.__str__()

class MonotoneTriangulator:
    """
    Corrected implementation of monotone polygon triangulation
    """
    
    def __init__(self) -> None:
        self.triangles = []
        self.diagonal_edges = []

    def triangulate(self, polygon, direction=None) -> List[Triangle]:
        """Main triangulation method"""
        self.triangles = []
        self.diagonal_edges = []
        
        vertices = polygon.vertices[:]
        n = len(vertices)
        
        if n < 3:
            return []
        if n == 3:
            self.triangles = [Triangle(vertices[0], vertices[1], vertices[2])]
            return self.triangles
        
        # Step 1: Sort vertices by x-coordinate (assuming x-monotone)
        sorted_vertices = sorted(vertices, key=lambda p: (p.x, p.y))
        
        # Step 2: Classify vertices into left and right chains
        leftmost = sorted_vertices[0]
        rightmost = sorted_vertices[-1]
        
        # Find chains by walking around the polygon
        left_chain, right_chain = self._find_chains(polygon, leftmost, rightmost)
        
        # Create vertex to chain mapping
        vertex_to_chain = {}
        for v in left_chain:
            vertex_to_chain[id(v)] = ChainType.LEFT
        for v in right_chain:
            vertex_to_chain[id(v)] = ChainType.RIGHT
        
        # The leftmost and rightmost vertices belong to both chains
        # Convention: leftmost = LEFT, rightmost = RIGHT
        vertex_to_chain[id(leftmost)] = ChainType.LEFT
        vertex_to_chain[id(rightmost)] = ChainType.RIGHT
        
        # Step 3: Triangulate using corrected stack algorithm
        return self._corrected_stack_triangulation(sorted_vertices, vertex_to_chain)

    def _find_chains(self, polygon, leftmost, rightmost) -> Tuple[List[Point], List[Point]]:
        """Find left and right chains of the polygon"""
        vertices = polygon.vertices
        n = len(vertices)
        
        # Find indices of leftmost and rightmost vertices
        left_idx = vertices.index(leftmost)
        right_idx = vertices.index(rightmost)
        
        # Split polygon into two chains
        if left_idx < right_idx:
            # Upper chain (left to right)
            upper_chain = vertices[left_idx:right_idx + 1]
            # Lower chain (right to left, reversed)
            lower_chain = vertices[right_idx:] + vertices[:left_idx + 1]
            lower_chain = list(reversed(lower_chain))
        else:
            # Upper chain (left to right)
            upper_chain = vertices[left_idx:] + vertices[:right_idx + 1]
            # Lower chain (right to left, reversed) 
            lower_chain = vertices[right_idx:left_idx + 1]
            lower_chain = list(reversed(lower_chain))
        
        # Determine which is left and which is right based on polygon orientation
        # For a counter-clockwise polygon, upper chain is typically the left chain
        return upper_chain, lower_chain

    def _corrected_stack_triangulation(self, sorted_vertices, vertex_to_chain) -> List[Triangle]:
        """Corrected stack-based triangulation algorithm"""
        n = len(sorted_vertices)
        stack = []
        
        # Initialize with first two vertices
        stack.append(sorted_vertices[0])
        stack.append(sorted_vertices[1])
        
        # Process vertices 2 to n-2
        for i in range(2, n):
            current = sorted_vertices[i]
            
            if i == n - 1:  # Last vertex
                # Connect to all vertices on stack (except the last one)
                while len(stack) > 1:
                    v2 = stack.pop()
                    v1 = stack[-1]
                    
                    # Create triangle with proper orientation
                    if self._is_valid_triangle(v1, v2, current):
                        triangle = Triangle(v1, v2, current)
                        self.triangles.append(triangle)
                        if len(stack) > 1:  # Don't add diagonal for last triangle
                            self.diagonal_edges.append((v1, current))
                    
            else:
                # Check if current vertex is on same chain as top of stack
                current_chain = vertex_to_chain[id(current)]
                top_chain = vertex_to_chain[id(stack[-1])]
                
                if current_chain == top_chain:
                    # Same chain: try to form triangles with stack vertices
                    self._process_same_chain(current, stack, vertex_to_chain)
                else:
                    # Different chain: connect to all stack vertices except last
                    self._process_different_chain(current, stack)
        
        return self.triangles

    def _process_same_chain(self, current, stack, vertex_to_chain) -> None:
        """Process vertex on same chain as stack top"""
        # Remove the top vertex from stack
        last_popped = stack.pop()
        
        # Try to form triangles with vertices on stack
        while len(stack) > 0:
            stack_top = stack[-1]
            
            # Check if we can form a valid triangle
            if self._can_triangulate(stack_top, last_popped, current, vertex_to_chain):
                # Form triangle
                triangle = Triangle(stack_top, last_popped, current)
                self.triangles.append(triangle)
                self.diagonal_edges.append((stack_top, current))
                
                # Continue with next vertex on stack
                last_popped = stack.pop()
            else:
                break
        
        # Push back the last popped vertex and current vertex
        stack.append(last_popped)
        stack.append(current)

    def _process_different_chain(self, current, stack) -> None:
        """Process vertex on different chain from stack top"""
        # Connect to all vertices on stack except the last one
        vertices_to_connect = stack[:-1]  # All except last
        
        for i in range(len(vertices_to_connect)):
            v1 = stack[i]
            v2 = stack[i + 1]
            
            # Form triangle
            if self._is_valid_triangle(v1, v2, current):
                triangle = Triangle(v1, v2, current)
                self.triangles.append(triangle)
                self.diagonal_edges.append((v1, current))
        
        # Keep only the last vertex from stack and add current
        last_vertex = stack[-1]
        stack.clear()
        stack.append(last_vertex)
        stack.append(current)

    def _can_triangulate(self, p1, p2, p3, vertex_to_chain) -> bool:
        """Check if three vertices can form a valid triangle in the triangulation"""
        # Basic validity check
        if not self._is_valid_triangle(p1, p2, p3):
            return False
        
        # For same chain vertices, check orientation
        chain = vertex_to_chain[id(p2)]
        
        # Calculate cross product to check orientation
        cross = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
        
        # For monotone polygon triangulation, we want consistent orientation
        if chain == ChainType.LEFT:
            return cross > 0  # Counter-clockwise for left chain
        else:
            return cross < 0  # Clockwise for right chain

    def _is_valid_triangle(self, p1, p2, p3) -> bool:
        """Check if three points form a valid (non-degenerate) triangle"""
        # Calculate area using cross product
        area = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
        return area > 1e-10  # Non-degenerate triangle

    def get_triangulation_stats(self) -> Dict[str, float | int | List[Triangle]]:
        """Get statistics about the triangulation"""
        if not self.triangles:
            return {}
        
        total_area = sum(triangle.area() for triangle in self.triangles)
        
        return {
            'num_triangles': len(self.triangles),
            'num_diagonals': len(self.diagonal_edges),
            'total_area': total_area,
            'average_triangle_area': total_area / len(self.triangles),
            'triangles': self.triangles,
            'diagonals': self.diagonal_edges
        }


class TriangulationVisualizer:
    """
    Visualization class for displaying triangulated monotone polygons
    """
    
    @staticmethod
    def visualize_triangulation(polygon: 'Polygon', triangles: List[Triangle], 
                              title: str = "Monotone Polygon Triangulation",
                              figsize: Tuple[int, int] = (14, 10),
                              show_stats: bool = True) -> None:
        """
        Create comprehensive visualization of triangulated polygon
        
        Args:
            polygon: Original polygon
            triangles: List of triangulation triangles
            title: Plot title
            figsize: Figure size
            show_stats: Whether to show triangulation statistics
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Original polygon
        TriangulationVisualizer._plot_original_polygon(ax1, polygon)
        ax1.set_title("Original Monotone Polygon", fontsize=14, fontweight='bold')
        
        # Right plot: Triangulated polygon
        TriangulationVisualizer._plot_triangulated_polygon(ax2, polygon, triangles)
        ax2.set_title("Triangulated Polygon", fontsize=14, fontweight='bold')
        
        # Add overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Add statistics if requested
        if show_stats and triangles:
            stats_text = f"Triangles: {len(triangles)}\n"
            stats_text += f"Vertices: {len(polygon.vertices)}\n"
            stats_text += f"Expected Triangles: {len(polygon.vertices) - 2}"
            
            fig.text(0.02, 0.95, stats_text, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_original_polygon(ax, polygon: 'Polygon'):
        """Plot the original polygon"""
        # Extract coordinates
        x_coords = [v.x for v in polygon.vertices]
        y_coords = [v.y for v in polygon.vertices]
        
        # Create polygon patch
        polygon_patch = MPLPolygon(list(zip(x_coords, y_coords)), 
                                 closed=True, 
                                 facecolor='lightblue', 
                                 edgecolor='navy', 
                                 linewidth=2,
                                 alpha=0.7)
        ax.add_patch(polygon_patch)
        
        # Plot vertices
        for i, vertex in enumerate(polygon.vertices):
            ax.plot(vertex.x, vertex.y, 'ro', markersize=8, zorder=5)
            ax.annotate(f'V{i}', (vertex.x, vertex.y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        TriangulationVisualizer._set_axis_limits(ax, polygon.vertices)
    
    @staticmethod
    def _plot_triangulated_polygon(ax, polygon: 'Polygon', triangles: List[Triangle]):
        """Plot the triangulated polygon with triangles highlighted"""
        # Plot original polygon boundary
        x_coords = [v.x for v in polygon.vertices]
        y_coords = [v.y for v in polygon.vertices]
        
        # Close the polygon for plotting
        x_coords_closed = x_coords + [x_coords[0]]
        y_coords_closed = y_coords + [y_coords[0]]
        
        ax.plot(x_coords_closed, y_coords_closed, 'k-', linewidth=3, alpha=0.8, label='Polygon Boundary')
        
        # Plot triangles with different colors
        cmap = plt.colormaps['Set3']
        colors = cmap(range(len(triangles)))

        for i, triangle in enumerate(triangles):
            # Triangle vertices
            tri_x = [p.x for p in triangle.vertices] + [triangle.vertices[0].x]
            tri_y = [p.y for p in triangle.vertices] + [triangle.vertices[0].y]
            
            # Fill triangle
            triangle_patch = MPLPolygon([(p.x, p.y) for p in triangle.vertices], 
                                      closed=True, 
                                      facecolor=colors[i], 
                                      edgecolor='red', 
                                      linewidth=1,
                                      alpha=0.6)
            ax.add_patch(triangle_patch)
            
            # Add triangle number at centroid
            centroid = triangle.centroid()
            ax.annotate(f'T{i+1}', (centroid.x, centroid.y), 
                       fontsize=10, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
        
        # Plot vertices
        for i, vertex in enumerate(polygon.vertices):
            ax.plot(vertex.x, vertex.y, 'ko', markersize=8, zorder=10)
            ax.annotate(f'V{i}', (vertex.x, vertex.y), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper right')
        TriangulationVisualizer._set_axis_limits(ax, polygon.vertices)
    
    @staticmethod
    def _set_axis_limits(ax, vertices: List['Point']):
        """Set appropriate axis limits with margin"""
        if not vertices:
            return
        
        margin = 0.5
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
    
    @staticmethod
    def visualize_step_by_step(polygon: 'Polygon', triangles: List[Triangle]):
        """Create step-by-step visualization of triangulation process"""
        n_steps = len(triangles)
        if n_steps == 0:
            print("No triangles to visualize")
            return
        
        # Create subplot grid
        cols = min(3, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]  
        elif rows == 1:
            axes = list(axes)  
        else:
            axes = axes.flatten()  
        
        # Plot each step
        for i in range(n_steps):
            ax = axes[i]
            
            # Plot polygon boundary
            x_coords = [v.x for v in polygon.vertices] + [polygon.vertices[0].x]
            y_coords = [v.y for v in polygon.vertices] + [polygon.vertices[0].y]
            ax.plot(x_coords, y_coords, 'k-', linewidth=2, alpha=0.8)
            
            # Plot triangles up to current step
            cmap = plt.get_cmap("Set3")
            colors = cmap(range(i+1))
            for j in range(i+1):
                triangle = triangles[j]
                triangle_patch = MPLPolygon([(p.x, p.y) for p in triangle.vertices], 
                                          closed=True, 
                                          facecolor=colors[j], 
                                          edgecolor='red', 
                                          linewidth=1,
                                          alpha=0.7)
                ax.add_patch(triangle_patch)
                
                # Add triangle label
                centroid = triangle.centroid()
                ax.annotate(f'T{j+1}', (centroid.x, centroid.y), 
                           fontsize=9, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
            
            # Plot vertices
            for k, vertex in enumerate(polygon.vertices):
                ax.plot(vertex.x, vertex.y, 'ko', markersize=6)
                ax.annotate(f'V{k}', (vertex.x, vertex.y), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_title(f'Step {i+1}: {i+1} Triangle{"s" if i > 0 else ""}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            TriangulationVisualizer._set_axis_limits(ax, polygon.vertices)
        
        # Hide unused subplots
        for i in range(n_steps, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle("Step-by-Step Triangulation Process", fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.show()



def create_sample_monotone_polygons() -> Dict[str, 'Polygon']:
    """Create sample monotone polygons for testing triangulation"""
    from core.monotonicitytester import Point, Polygon, MonotonicityTester

    # X-monotone polygon (mountain-like shape)
    x_monotone_polygon = Polygon([
        Point(0, 0), Point(2, -1), Point(4, 0), 
        Point(4, 2), Point(2, 3), Point(0, 2)
    ])
    
    # Y-monotone polygon
    y_monotone_polygon = Polygon([
        Point(0, 0), Point(-1, 2), Point(0, 4),
        Point(2, 4), Point(3, 2), Point(2, 0)
    ])
    
    # Simple x-monotone trapezoid
    trapezoid = Polygon([
        Point(0, 0), Point(1, 2), Point(3, 2), Point(4, 0)
    ])
    
    # Complex x-monotone polygon
    complex_x_monotone = Polygon([
        Point(0, 2), Point(1, 4), Point(2, 1), Point(3, 5),
        Point(4, 3), Point(5, 6), Point(7, 0),
        Point(5, -1), Point(4, 0.5), Point(3, -0.5), Point(2, 0),
        Point(1, 0.5)
    ])

    return {
        'x_monotone': x_monotone_polygon,
        #'y_monotone': y_monotone_polygon,
        'trapezoid': trapezoid,
        #'complex': complex_x_monotone
    }


def demonstrate_triangulation() -> None:
    """
    Demonstration function showing triangulation of various monotone polygons
    
    Algorithm Complexity Analysis:
    - Time Complexity: O(n) where n is the number of vertices
      * Sorting vertices: O(n log n) in general case, but for monotone polygons
        with proper preprocessing this can be O(n)
      * Chain identification: O(n)
      * Stack-based triangulation: O(n) - each vertex pushed/popped once
      
    - Space Complexity: O(n)
      * Stack storage: O(n) in worst case
      * Output triangles: O(n) triangles for n-vertex polygon
      * Additional data structures: O(n)
    
    The algorithm is optimal for monotone polygon triangulation.
    """
    print("=" * 80)
    print("MONOTONE POLYGON TRIANGULATION DEMONSTRATION")
    print("=" * 80)
    print("\nAlgorithm: Stack-based triangulation for monotone polygons")
    print("Time Complexity: O(n) where n = number of vertices")
    print("Space Complexity: O(n)")
    print("\nTheorem: Any monotone polygon with n vertices can be triangulated")
    print("into exactly (n-2) triangles using (n-3) diagonals.\n")
    
    # Create sample polygons
    sample_polygons = create_sample_monotone_polygons()
    triangulator = MonotoneTriangulator()
    
    for name, polygon in sample_polygons.items():
        print(f"\nTriangulating {name.upper()} polygon:")
        print(f"Vertices: {len(polygon.vertices)}")
        print(f"Expected triangles: {len(polygon.vertices) - 2}")
        
        try:
            # Perform triangulation
            triangles = triangulator.triangulate(polygon)

            
            # Get statistics
            stats = triangulator.get_triangulation_stats()
            
            print(f"✓ Triangulation successful!")
            print(f"  - Triangles generated: {stats['num_triangles']}")
            print(f"  - Diagonals added: {stats['num_diagonals']}")
            print(f"  - Total area: {stats['total_area']:.3f}")
            
            # Visualize
            TriangulationVisualizer.visualize_triangulation(
                polygon, triangles, 
                title=f"Triangulation of {name.title()} Polygon",
                show_stats=True
            )
            
            # Step-by-step visualization for smaller polygons
            if len(triangles) <= 6:
                TriangulationVisualizer.visualize_step_by_step(polygon, triangles)
            
        except Exception as e:
            print(f"✗ Triangulation failed: {e}")


if __name__ == "__main__":
    # Import required classes from the main module
    try:
        from monotonicitytester import Point, Polygon
        demonstrate_triangulation()
    except ImportError:
        print("Error: Could not import required classes from monotonicitytester.py")
        print("Please ensure this script is run with access to Point, Polygon, and Direction classes.")