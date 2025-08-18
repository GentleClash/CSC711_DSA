from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


class Direction(Enum):
    """Direction enumeration for monotonicity testing"""
    X_DIRECTION = "x"
    Y_DIRECTION = "y"
    CUSTOM = "custom"


class Point:
    """Represents a 2D point with x, y coordinates"""
    
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Polygon:
    """Represents a polygon with vertices and provides monotonicity testing"""
    
    def __init__(self, vertices: List[Point]):
        """
        Initialize polygon with vertices
        
        Args:
            vertices: List of Point objects representing polygon vertices
                     Should be in order (clockwise or counterclockwise)
        """
        if len(vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices")
        
        self.vertices = vertices[:]
        self._ensure_simple_polygon()

    def _ensure_simple_polygon(self) -> None:
        """Ensure vertices form a simple (non-self-intersecting) polygon"""
        # Remove duplicate consecutive points
        cleaned_vertices = []
        for i, vertex in enumerate(self.vertices):
            next_vertex = self.vertices[(i + 1) % len(self.vertices)]
            if vertex != next_vertex:
                cleaned_vertices.append(vertex)
        
        if len(cleaned_vertices) < 3:
            raise ValueError("Polygon has insufficient distinct vertices")
        
        self.vertices = cleaned_vertices
    
    def is_x_monotone(self) -> bool:
        """Test if polygon is monotone with respect to x-axis"""
        return self._is_monotone_direction(Direction.X_DIRECTION)
    
    def is_y_monotone(self) -> bool:
        """Test if polygon is monotone with respect to y-axis"""
        return self._is_monotone_direction(Direction.Y_DIRECTION)
    
    def is_monotone_custom_direction(self, direction_vector: Tuple[float, float]) -> bool:
        """
        Test if polygon is monotone with respect to a custom direction
        
        Args:
            direction_vector: Tuple (dx, dy) representing the direction
        
        Returns:
            bool: True if polygon is monotone in the given direction
        """
        return self._is_monotone_direction(Direction.CUSTOM, direction_vector)
    
    def _is_monotone_direction(self, direction: Direction, 
                             direction_vector: Optional[Tuple[float, float]] = None) -> bool:
        """
        Core monotonicity testing algorithm
        
        A polygon is monotone with respect to a line L if:
        1. Every line perpendicular to L intersects the polygon at most twice
        2. The boundary can be split into two monotone chains
        """
        if direction == Direction.X_DIRECTION:
            # For x-monotone: sort by x-coordinate
            sorted_vertices = sorted(self.vertices, key=lambda p: p.x)
            coord_func = lambda p: p.x
        elif direction == Direction.Y_DIRECTION:
            # For y-monotone: sort by y-coordinate
            sorted_vertices = sorted(self.vertices, key=lambda p: p.y)
            coord_func = lambda p: p.y
        else:
            # Custom direction
            if direction_vector is None:
                raise ValueError("Direction vector required for custom direction")
            
            dx, dy = direction_vector
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                raise ValueError("Direction vector cannot be zero")
            
            # Project points onto the direction vector
            def projection(p) -> float:
                return p.x * dx + p.y * dy
            
            sorted_vertices = sorted(self.vertices, key=projection)
            coord_func = projection
        
        # Find extreme points (min and max in the chosen direction)
        min_vertex = sorted_vertices[0]
        max_vertex = sorted_vertices[-1]
        
        # Split polygon into two chains from min to max vertex
        try:
            upper_chain, lower_chain = self._split_into_chains(min_vertex, max_vertex)
        except ValueError:
            return False
        
        # Check if both chains are monotone in the chosen direction
        return (self._is_chain_monotone(upper_chain, coord_func) and 
                self._is_chain_monotone(lower_chain, coord_func))
    
    def _split_into_chains(self, start_vertex: Point, end_vertex: Point) -> Tuple[List[Point], List[Point]]:
        """
        Split polygon boundary into two chains from start to end vertex
        
        Returns:
            Tuple of (upper_chain, lower_chain)
        """
        # Find indices of start and end vertices
        start_idx = None
        end_idx = None
        
        for i, vertex in enumerate(self.vertices):
            if vertex == start_vertex:
                start_idx = i
            if vertex == end_vertex:
                end_idx = i
        
        if start_idx is None or end_idx is None:
            raise ValueError("Start or end vertex not found in polygon")
        
        # Create two chains
        if start_idx <= end_idx:
            chain1 = self.vertices[start_idx:end_idx + 1]
            chain2 = self.vertices[end_idx:] + self.vertices[:start_idx + 1]
        else:
            chain1 = self.vertices[start_idx:] + self.vertices[:end_idx + 1]
            chain2 = self.vertices[end_idx:start_idx + 1]
        
        return chain1, chain2
    
    def _is_chain_monotone(self, chain: List[Point], coord_func) -> bool:
        """
        Check if a chain of vertices is monotone with respect to coordinate function
        
        Args:
            chain: List of points forming the chain
            coord_func: Function to extract coordinate value for comparison
        
        Returns:
            bool: True if chain is monotone
        """
        if len(chain) <= 2:
            return True
        
        # Check if chain is monotone (either increasing or decreasing)
        coords = [coord_func(point) for point in chain]
        
        # Check for non-decreasing sequence
        is_non_decreasing = all(coords[i] <= coords[i + 1] for i in range(len(coords) - 1))
        
        # Check for non-increasing sequence
        is_non_increasing = all(coords[i] >= coords[i + 1] for i in range(len(coords) - 1))
        
        return is_non_decreasing or is_non_increasing
    
    def get_bounding_box(self) -> Tuple[Point, Point]:
        """
        Get bounding box of the polygon
        
        Returns:
            Tuple of (bottom_left, top_right) points
        """
        min_x = min(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        
        return Point(min_x, min_y), Point(max_x, max_y)
    

    def visualize(self, title: Optional[str] = None, show_monotonicity: bool = True, 
                  figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Enhanced visualization of the polygon with monotonicity information
        
        Args:
            title: Custom title for the plot
            show_monotonicity: Whether to display monotonicity test results
            figsize: Figure size tuple (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MPLPolygon
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        # Create figure with subplots
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Extract coordinates
        x_coords = [v.x for v in self.vertices]
        y_coords = [v.y for v in self.vertices]
        
        # Close the polygon for proper visualization
        x_coords_closed = x_coords + [x_coords[0]]
        y_coords_closed = y_coords + [y_coords[0]]
        
        # Create polygon patch with nice styling
        polygon_patch = MPLPolygon(list(zip(x_coords, y_coords)), 
                                 closed=True, 
                                 facecolor='lightblue', 
                                 edgecolor='navy', 
                                 linewidth=2,
                                 alpha=0.7)
        ax.add_patch(polygon_patch)
        
        # Plot vertices with numbers
        for i, vertex in enumerate(self.vertices):
            ax.plot(vertex.x, vertex.y, 'ro', markersize=8, zorder=5)
            ax.annotate(f'V{i}', (vertex.x, vertex.y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot edges with direction arrows
        for i in range(len(self.vertices)):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % len(self.vertices)]
            
            # Draw edge
            ax.plot([start.x, end.x], [start.y, end.y], 'b-', linewidth=2, alpha=0.8)
            
            # Add direction arrow
            mid_x = (start.x + end.x) / 2
            mid_y = (start.y + end.y) / 2
            dx = end.x - start.x
            dy = end.y - start.y
            
            # Normalize and scale arrow
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx_norm = dx / length * 0.3
                dy_norm = dy / length * 0.3
                ax.arrow(mid_x - dx_norm/2, mid_y - dy_norm/2, dx_norm, dy_norm,
                        head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        # Add grid and styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            ax.set_title(f'Polygon Visualization ({len(self.vertices)} vertices)', 
                        fontsize=16, fontweight='bold', pad=20)
        
        # Add monotonicity information
        if show_monotonicity:
            monotonicity_info = []
            
            try:
                x_mono = self.is_x_monotone()
                monotonicity_info.append(f"X-monotone: {'✓' if x_mono else '✗'}")
            except:
                monotonicity_info.append("X-monotone: Error")
            
            try:
                y_mono = self.is_y_monotone()
                monotonicity_info.append(f"Y-monotone: {'✓' if y_mono else '✗'}")
            except:
                monotonicity_info.append("Y-monotone: Error")
            
            # Add text box with monotonicity results
            info_text = '\n'.join(monotonicity_info)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        # Add coordinate labels
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Adjust margins
        margin = 0.5
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_with_chains(self, direction: Direction = Direction.X_DIRECTION) -> None:
        """
        Visualize polygon with monotone chains highlighted
        
        Args:
            direction: Direction for chain decomposition (X_DIRECTION or Y_DIRECTION)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import Polygon as MPLPolygon
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Extract coordinates
        x_coords = [v.x for v in self.vertices]
        y_coords = [v.y for v in self.vertices]
        
        # Find extreme points based on direction
        if direction == Direction.X_DIRECTION:
            sorted_vertices = sorted(self.vertices, key=lambda p: p.x)
            direction_name = "X"
        else:  # Y_DIRECTION
            sorted_vertices = sorted(self.vertices, key=lambda p: p.y)
            direction_name = "Y"
        
        min_vertex = sorted_vertices[0]
        max_vertex = sorted_vertices[-1]
        
        try:
            # Get chains
            upper_chain, lower_chain = self._split_into_chains(min_vertex, max_vertex)
            
            # Plot polygon outline
            polygon_patch = MPLPolygon(list(zip(x_coords, y_coords)), 
                                     closed=True, 
                                     facecolor='lightgray', 
                                     edgecolor='black', 
                                     linewidth=1,
                                     alpha=0.3)
            ax.add_patch(polygon_patch)
            
            # Plot upper chain in blue
            if len(upper_chain) > 1:
                upper_x = [p.x for p in upper_chain]
                upper_y = [p.y for p in upper_chain]
                ax.plot(upper_x, upper_y, 'b-', linewidth=4, alpha=0.8, label='Upper Chain')
                
                # Mark chain points
                for i, p in enumerate(upper_chain):
                    ax.plot(p.x, p.y, 'bo', markersize=8)
                    ax.annotate(f'U{i}', (p.x, p.y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=9)
            
            # Plot lower chain in red
            if len(lower_chain) > 1:
                lower_x = [p.x for p in lower_chain]
                lower_y = [p.y for p in lower_chain]
                ax.plot(lower_x, lower_y, 'r-', linewidth=4, alpha=0.8, label='Lower Chain')
                
                # Mark chain points
                for i, p in enumerate(lower_chain):
                    ax.plot(p.x, p.y, 'ro', markersize=8)
                    ax.annotate(f'L{i}', (p.x, p.y), xytext=(5, -15), 
                              textcoords='offset points', fontsize=9)
            
            # Highlight extreme points
            ax.plot(min_vertex.x, min_vertex.y, 'go', markersize=12, 
                   label=f'Min {direction_name}')
            ax.plot(max_vertex.x, max_vertex.y, 'mo', markersize=12, 
                   label=f'Max {direction_name}')
            
            # Check if chains are monotone
            coord_func = (lambda p: p.x) if direction == Direction.X_DIRECTION else (lambda p: p.y)
            upper_mono = self._is_chain_monotone(upper_chain, coord_func)
            lower_mono = self._is_chain_monotone(lower_chain, coord_func)
            
            chain_info = f"Upper Chain Monotone: {'✓' if upper_mono else '✗'}\n"
            chain_info += f"Lower Chain Monotone: {'✓' if lower_mono else '✗'}\n"
            chain_info += f"Overall {direction_name}-Monotone: {'✓' if upper_mono and lower_mono else '✗'}"
            
        except Exception as e:
            chain_info = f"Error in chain decomposition: {str(e)}"
        
        # Styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Polygon Chain Decomposition ({direction_name}-direction)', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        
        # Add chain information
        ax.text(0.02, 0.98, chain_info, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        # Set labels and margins
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        margin = 0.5
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        return f"Polygon with {len(self.vertices)} vertices: {self.vertices}"
    

class MonotonicityTester:
    """Main class for testing polygon monotonicity with comprehensive reporting"""
    
    @staticmethod
    def test_polygon_monotonicity(polygon: Polygon) -> dict:
        """
        Comprehensive test of polygon monotonicity
        
        Args:
            polygon: Polygon instance to test
        
        Returns:
            dict: Test results with detailed information
        """
        results = {
            'polygon': str(polygon),
            'vertex_count': len(polygon.vertices),
            'bounding_box': polygon.get_bounding_box(),
            'monotonicity_tests': {}
        }
        
        # Test x-monotonicity
        try:
            x_monotone = polygon.is_x_monotone()
            results['monotonicity_tests']['x_monotone'] = {
                'result': x_monotone,
                'description': 'Monotone with respect to x-axis' if x_monotone else 'Not x-monotone'
            }
        except Exception as e:
            results['monotonicity_tests']['x_monotone'] = {
                'result': False,
                'error': str(e)
            }
        
        # Test y-monotonicity
        try:
            y_monotone = polygon.is_y_monotone()
            results['monotonicity_tests']['y_monotone'] = {
                'result': y_monotone,
                'description': 'Monotone with respect to y-axis' if y_monotone else 'Not y-monotone'
            }
        except Exception as e:
            results['monotonicity_tests']['y_monotone'] = {
                'result': False,
                'error': str(e)
            }
        
        # Test custom direction (diagonal)
        try:
            diagonal_monotone = polygon.is_monotone_custom_direction((1, 1))
            results['monotonicity_tests']['diagonal_monotone'] = {
                'result': diagonal_monotone,
                'description': 'Monotone with respect to diagonal (1,1)' if diagonal_monotone else 'Not diagonal-monotone'
            }
        except Exception as e:
            results['monotonicity_tests']['diagonal_monotone'] = {
                'result': False,
                'error': str(e)
            }
        
        return results
    
    @staticmethod
    def print_test_results(results: dict):
        """Print formatted test results"""
        print(f"\n{'='*60}")
        print("POLYGON MONOTONICITY TEST RESULTS")
        print(f"{'='*60}")
        print(f"Polygon: {results['polygon']}")
        print(f"Vertex Count: {results['vertex_count']}")
        
        bbox = results['bounding_box']
        print(f"Bounding Box: {bbox[0]} to {bbox[1]}")
        
        print(f"\n{'Monotonicity Tests:':<25}")
        print(f"{'-'*40}")
        
        for test_name, test_data in results['monotonicity_tests'].items():
            status = "✓ PASS" if test_data['result'] else "✗ FAIL"
            print(f"{test_name:<20} {status:<8}")
            
            if 'description' in test_data:
                print(f"{'':20} {test_data['description']}")
            elif 'error' in test_data:
                print(f"{'':20} Error: {test_data['error']}")


def create_test_polygons() -> Dict[str, Polygon]:
    """Create test polygons for demonstration"""
    
    # X-monotone polygon (rectangle-like)
    x_monotone_polygon = Polygon([
        Point(0, 0), Point(2, -1), Point(4, 0), 
        Point(4, 2), Point(2, 3), Point(0, 2)
    ])
    
    # Y-monotone polygon
    y_monotone_polygon = Polygon([
        Point(0, 0), Point(-1, 2), Point(0, 4),
        Point(2, 4), Point(3, 2), Point(2, 0)
    ])
    
    # Non-monotone polygon (star-like shape)
    non_monotone_polygon = Polygon([
        Point(0, 2), Point(1, 1), Point(2, 2),
        Point(1, 3), Point(2, 4), Point(0, 3),
        Point(-2, 4), Point(-1, 3), Point(-2, 2),
        Point(-1, 1)
    ])
    
    # Simple triangle (should be both x and y monotone)
    triangle = Polygon([
        Point(0, 0), Point(2, 0), Point(1, 2)
    ])
    
    return {
        'x_monotone': x_monotone_polygon,
        'y_monotone': y_monotone_polygon,
        'non_monotone': non_monotone_polygon,
        'triangle': triangle
    }


def main() -> None:
    """Main function demonstrating the monotonicity checker"""
    print("Polygon Monotonicity Checker")
    print("Testing various polygon types for monotonicity\n")
    
    # Create test polygons
    test_polygons = create_test_polygons()
    
    tester = MonotonicityTester()
    
    # Test each polygon
    for polygon_name, polygon in test_polygons.items():
        print(f"\nTesting {polygon_name.upper()} polygon:")
        results = tester.test_polygon_monotonicity(polygon)
        tester.print_test_results(results)
        #polygon.visualize(title=f"{polygon_name.capitalize()} Polygon", show_monotonicity=True)
        #polygon.visualize_with_chains(Direction.X_DIRECTION)
        #polygon.visualize_with_chains(Direction.Y_DIRECTION)

    # Example of custom direction testing
    print(f"\n{'='*60}")
    print("CUSTOM DIRECTION TESTING")
    print(f"{'='*60}")
    
    x_monotone = test_polygons['x_monotone']
    x_monotone.visualize(title="X-Monotone Polygon", show_monotonicity=True)
    x_monotone.visualize_with_chains(Direction.X_DIRECTION)
    x_monotone.visualize_with_chains(Direction.Y_DIRECTION)

    # Test various custom directions
    custom_directions = [
        (1, 0),   # Pure x direction
        (0, 1),   # Pure y direction
        (1, 1),   # Diagonal
        (1, -1),  # Negative diagonal
        (3, 4)    # Arbitrary direction
    ]
    
    for direction in custom_directions:
        is_monotone = x_monotone.is_monotone_custom_direction(direction)
        status = "✓" if is_monotone else "✗"
        print(f"Direction {direction}: {status} {'Monotone' if is_monotone else 'Not monotone'}")

    """non_monotone_polygon = test_polygons['non_monotone']
    non_monotone_polygon.visualize(title="Non-Monotone Polygon", show_monotonicity=True)
    non_monotone_polygon.visualize_with_chains(Direction.X_DIRECTION)
    non_monotone_polygon.visualize_with_chains(Direction.Y_DIRECTION)

    # Test various custom directions
    custom_directions = [
        (1, 0),   # Pure x direction
        (0, 1),   # Pure y direction
        (1, 1),   # Diagonal
        (1, -1),  # Negative diagonal
        (3, 4)    # Arbitrary direction
    ]

    for direction in custom_directions:
        is_monotone = triangle.is_monotone_custom_direction(direction)
        status = "✓" if is_monotone else "✗"
        print(f"Direction {direction}: {status} {'Monotone' if is_monotone else 'Not monotone'}")"""

if __name__ == "__main__":
    main()