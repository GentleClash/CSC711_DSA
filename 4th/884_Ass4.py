from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class Point:
    """Represents a point in 2D space."""
    x: float
    y: float
    
    def __getitem__(self, index: int) -> float:
        """Allow indexing like point[0] for x, point[1] for y."""
        return self.x if index == 0 else self.y
    
    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


@dataclass
class KDNode:
    """Node in a KD-tree."""
    point: Point
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None
    axis: int = 0  # 0 for x-axis, 1 for y-axis
    
    def __repr__(self) -> str:
        axis_name = 'x' if self.axis == 0 else 'y'
        return f"Node({self.point}, axis={axis_name})"


class KDTree:
    """
    KD-tree implementation for 2D points.
    
    Features:
    - Recursive build and search functions
    - Visualization capabilities
    - Range query support
    - Detailed logging of partitions
    """
    
    def __init__(self) -> None:
        self.root: Optional[KDNode] = None
        self.build_log: List[str] = []
    
    def build_kdtree(self, points: List[Union[List[float], Tuple[float, float], Point]], 
                     depth: int = 0, parent_info: str = "Root") -> Optional[KDNode]:
        """
        Recursively build a 2D KD-tree.
        
        Args:
            points: List of points to build tree from
            depth: Current depth (determines splitting axis)
            parent_info: Information about parent for logging
            
        Returns:
            Root node of the constructed subtree
        """
        if not points:
            return None
        
        # Convert input to Point objects if needed
        point_objects = []
        for p in points:
            if isinstance(p, Point):
                point_objects.append(p)
            else:
                point_objects.append(Point(p[0], p[1]))
        
        # Determine splitting axis (0 for x, 1 for y)
        axis = depth % 2
        axis_name = 'x' if axis == 0 else 'y'
        
        # Sort points by the current axis
        point_objects.sort(key=lambda point: point[axis])
        
        # Find median
        median_idx = len(point_objects) // 2
        median_point = point_objects[median_idx]
        
        # Log the partition
        log_msg = (f"Depth {depth} ({parent_info}): Splitting on {axis_name}-axis, "
                  f"Median: {median_point}, Points: {point_objects}")
        self.build_log.append(log_msg)
        print(log_msg)
        
        # Create node
        node = KDNode(point=median_point, axis=axis)
        
        # Recursively build left and right subtrees
        left_points = point_objects[:median_idx]
        right_points = point_objects[median_idx + 1:]
        
        if left_points:
            node.left = self.build_kdtree(left_points, depth + 1, 
                                        f"Left of {median_point}")
        
        if right_points:
            node.right = self.build_kdtree(right_points, depth + 1, 
                                         f"Right of {median_point}")
        
        return node
    
    def build(self, points: List[Union[List[float], Tuple[float, float], Point]]) -> None:
        """Build the KD-tree from a list of points."""
        self.build_log.clear()
        print("=== Building KD-Tree ===")
        self.root = self.build_kdtree(points)
        print("=== Build Complete ===\n")
    
    def search_kdtree(self, nu: Optional[KDNode], search_range: List[List[float]], 
                     depth: int = 0, query_point: Optional[Point] = None) -> List[Point]:
        """
        SEARCHKDTREE(ν,R) - Formal KD-tree range search algorithm
        
        Args:
            nu: The root of (a subtree of) a kd-tree (corresponds to ν in algorithm)
            search_range: Range R - Rectangle [[x_min, y_min], [x_max, y_max]]
            depth: Current depth (for axis determination)
            query_point: Reference point to check if found points are "below" it
            
        Returns:
            All points at leaves below ν that lie in the range R and are below query_point
        """
        # Base case: if subtree is empty
        if nu is None:
            return []
        
        results = []
        axis = depth % 2
        point = nu.point
        x_min, y_min = search_range[0]
        x_max, y_max = search_range[1]
        
        print(f"Visiting node {point} (axis={'x' if axis == 0 else 'y'})")
        
        # Check if current point lies in the range
        in_range = (x_min <= point.x <= x_max and y_min <= point.y <= y_max)
        
        if in_range:
            # Additional check: is this point "below" the query point?
            if query_point is None:
                results.append(point)
                print(f"  ✓ Point {point} is in range R")
            else:
                # Check if point is "below" query_point (lower y-coordinate)
                if point.y <= query_point.y:
                    results.append(point)
                    print(f"  ✓ Point {point} is in range R and below query point {query_point}")
                else:
                    print(f"  ✗ Point {point} is in range R but NOT below query point {query_point}")
        else:
            print(f"  ✗ Point {point} is outside range R")
        
        # Determine which subtrees to search based on range intersection
        if axis == 0:  # x-axis split (vertical line at point.x)
            print(f"  Checking x-axis split at x = {point.x}")
            
            # Search left subtree if range intersects left half-space
            if x_min <= point.x:
                print(f"  → Range extends left of split (x_min={x_min} ≤ {point.x})")
                left_results = self.search_kdtree(nu.left, search_range, depth + 1, query_point)
                results.extend(left_results)
            else:
                print(f"  ✗ Range doesn't extend left of split")
            
            # Search right subtree if range intersects right half-space  
            if x_max >= point.x:
                print(f"  → Range extends right of split (x_max={x_max} ≥ {point.x})")
                right_results = self.search_kdtree(nu.right, search_range, depth + 1, query_point)
                results.extend(right_results)
            else:
                print(f"  ✗ Range doesn't extend right of split")
                
        else:  # y-axis split (horizontal line at point.y)
            print(f"  Checking y-axis split at y = {point.y}")
            
            # Search left subtree if range intersects lower half-space
            if y_min <= point.y:
                print(f"  → Range extends below split (y_min={y_min} ≤ {point.y})")
                left_results = self.search_kdtree(nu.left, search_range, depth + 1, query_point)
                results.extend(left_results)
            else:
                print(f"  ✗ Range doesn't extend below split")
            
            # Search right subtree if range intersects upper half-space
            if y_max >= point.y:
                print(f"  → Range extends above split (y_max={y_max} ≥ {point.y})")
                right_results = self.search_kdtree(nu.right, search_range, depth + 1, query_point)
                results.extend(right_results)
            else:
                print(f"  ✗ Range doesn't extend above split")
        
        return results
    
    def range_search(self, search_range: List[List[float]], 
                    start_node: Optional[KDNode] = None,
                    query_point: Optional[Union[List[float], Point]] = None) -> List[Point]:
        """
        SEARCHKDTREE(ν,R) - Search for all points in subtree ν that lie in range R.
        
        Args:
            search_range: Range R - Rectangle [[x_min, y_min], [x_max, y_max]]  
            start_node: Starting node ν (uses root if None)
            query_point: Reference point - only return points "below" this point
            
        Returns:
            All points at leaves below ν that lie in the range R (and below query_point if provided)
        """
        if start_node is None:
            start_node = self.root
            
        # Convert query_point to Point object if provided
        query_pt = None
        if query_point is not None:
            if isinstance(query_point, list):
                query_pt = Point(query_point[0], query_point[1])
            else:
                query_pt = query_point
        
        nu_desc = start_node.point if start_node else 'None'
        constraint_desc = f", with points below {query_pt}" if query_pt else ""
        print(f"\n=== SEARCHKDTREE(ν={nu_desc}, R={search_range}{constraint_desc}) ===")
        
        results = self.search_kdtree(start_node, search_range, query_point=query_pt)
        
        print(f"=== Search complete. Found {len(results)} points: {results} ===\n")
        return results
    
    def print_tree(self) -> None:
        """
        Print a visual representation of the KD-tree using PrettyPrintTree.
        Falls back to manual printing if PrettyPrintTree is not available.
        """
        try:
            from PrettyPrint import PrettyPrintTree
            if self.root is None:
                print("Tree is empty")
                return
            
            get_children = lambda node: [child for child in [node.left, node.right] if child is not None]
            
            get_value = lambda node: f"{node.point}{'x' if node.axis == 0 else 'y'}"
            
            pt = PrettyPrintTree(get_children, get_value) # type: ignore
            pt(self.root) # type: ignore
        except ImportError:
            self.print_tree_fallback(self.root)



    def print_tree_fallback(self, node: Optional[KDNode] = None, level: int = 0, 
                   prefix: str = "Root: ") -> None:
        """
        Print a visual representation of the tree.
        
        Args:
            node: Starting node (uses root if None)
            level: Current level (for indentation)
            prefix: Prefix string for the current node
        """
        if node is None:
            node = self.root
        
        if node is not None:
            axis_name = 'x' if node.axis == 0 else 'y'
            print(" " * (level * 4) + prefix + f"{node.point} (split on {axis_name})")
            
            if node.left is not None or node.right is not None:
                if node.left:
                    self.print_tree_fallback(node.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- None")
                
                if node.right:
                    self.print_tree_fallback(node.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- None")
    
    def visualize(self, query_ranges: Optional[List[List[List[float]]]] = None,
                  figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize the KD-tree with points and optional search ranges.
        
        Args:
            query_ranges: List of search ranges to highlight
            figsize: Figure size for the plot
        """
        if self.root is None:
            print("Tree is empty, cannot visualize.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Collect all points for plotting
        points = self._collect_points(self.root)
        
        if not points:
            print("No points to visualize.")
            return
        
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        # Plot 1: Points with partitioning lines
        ax1.scatter(x_coords, y_coords, c='red', s=50, zorder=5)
        
        # Add point labels
        for point in points:
            ax1.annotate(f'{point}', (point.x, point.y), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Draw partitioning lines
        x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
        y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
        
        self._draw_partitions(ax1, self.root, x_min, x_max, y_min, y_max)
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('KD-Tree Partitioning')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Plot 2: Tree structure
        ax2.axis('off')
        ax2.set_title('Tree Structure')
        
        # Simple tree visualization
        if self.root:
            self._draw_tree_structure(ax2, self.root, 0.5, 0.9, 0.4, 0)
        
        # Add search ranges if provided
        if query_ranges:
            for i, search_range in enumerate(query_ranges):
                x_min_r, y_min_r = search_range[0]
                x_max_r, y_max_r = search_range[1]
                width = x_max_r - x_min_r
                height = y_max_r - y_min_r
                
                rect = patches.Rectangle((x_min_r, y_min_r), width, height,
                                       linewidth=2, edgecolor=f'C{i}', 
                                       facecolor=f'C{i}', alpha=0.2,
                                       label=f'Range {i+1}')
                ax1.add_patch(rect)
        
        if query_ranges:
            ax1.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _collect_points(self, node: Optional[KDNode]) -> List[Point]:
        """Collect all points in the tree."""
        if node is None:
            return []
        
        points = [node.point]
        points.extend(self._collect_points(node.left))
        points.extend(self._collect_points(node.right))
        return points
    
    def _draw_partitions(self, ax, node: Optional[KDNode], 
                        x_min: float, x_max: float, 
                        y_min: float, y_max: float, depth: int = 0) -> None:
        """Draw partitioning lines recursively."""
        if node is None:
            return
        
        axis = depth % 2
        
        if axis == 0:  # Vertical line (x-axis split)
            ax.axvline(x=node.point.x, ymin=(y_min - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                      ymax=(y_max - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                      color='blue', linestyle='--', alpha=0.7, linewidth=1)
            
            # Recurse on left and right with updated x bounds
            self._draw_partitions(ax, node.left, x_min, node.point.x, y_min, y_max, depth + 1)
            self._draw_partitions(ax, node.right, node.point.x, x_max, y_min, y_max, depth + 1)
        
        else:  # Horizontal line (y-axis split)
            ax.axhline(y=node.point.y, xmin=(x_min - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                      xmax=(x_max - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                      color='green', linestyle='--', alpha=0.7, linewidth=1)
            
            # Recurse on left and right with updated y bounds
            self._draw_partitions(ax, node.left, x_min, x_max, y_min, node.point.y, depth + 1)
            self._draw_partitions(ax, node.right, x_min, x_max, node.point.y, y_max, depth + 1)
    
    def _draw_tree_structure(self, ax, node: Optional[KDNode], x: float, y: float, 
                           width: float, depth: int) -> None:
        """Draw the tree structure as a hierarchical diagram."""
        if node is None:
            return
        
        # Draw current node
        axis_name = 'x' if node.axis == 0 else 'y'
        circle = patches.Circle((x, y), 0.03, color='lightblue', zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f'{node.point}\n({axis_name})', ha='center', va='center', 
                fontsize=8, zorder=4)
        
        # Draw connections and recurse
        if node.left or node.right:
            child_width = width / 2
            child_y = y - 0.15
            
            if node.left:
                left_x = x - width / 4
                ax.plot([x, left_x], [y - 0.03, child_y + 0.03], 'k-', alpha=0.6)
                self._draw_tree_structure(ax, node.left, left_x, child_y, 
                                        child_width, depth + 1)
            
            if node.right:
                right_x = x + width / 4
                ax.plot([x, right_x], [y - 0.03, child_y + 0.03], 'k-', alpha=0.6)
                self._draw_tree_structure(ax, node.right, right_x, child_y, 
                                        child_width, depth + 1)


def test_kdtree():
    """Test the KD-tree implementation with the provided sample data."""
    
    # Sample input data
    sample_points = [
        [1, 9], [2, 3], [4, 1], [3, 7], [5, 4], 
        [6, 8], [7, 2], [8, 8], [7, 9], [9, 6]
    ]
    
    print("Sample Points:", sample_points)
    print()
    
    # Build the KD-tree
    kdtree = KDTree()
    kdtree.build(sample_points)
    
    # Print tree structure
    print("=== Tree Structure ===")
    kdtree.print_tree()
    print()
    
    # Test range searches
    test_cases = [
        {
            'v': [2, 3],
            'R': [[3, 7], [7, 9]],
            'description': 'Original test case'
        },
        {
            'v': [5, 5],
            'R': [[1, 1], [6, 8]],
            'description': 'Lower left quadrant'
        },
        {
            'v': [7, 7],
            'R': [[6, 6], [9, 9]],
            'description': 'Upper right region'
        },
        {
            'v': [4, 4],
            'R': [[0, 0], [10, 10]],
            'description': 'Entire space'
        }
    ]
    
    search_ranges = []
    
    # Test range searches with the formal algorithm
    test_ranges = [
        {
            'R': [[3, 7], [7, 9]], 
            'description': 'Original test case - upper right region'
        },
        {
            'R': [[1, 1], [6, 8]], 
            'description': 'Lower left quadrant'
        },
        {
            'R': [[6, 6], [9, 9]], 
            'description': 'Upper right region'
        },
        {
            'R': [[0, 0], [10, 10]], 
            'description': 'Entire space'
        }
    ]
    
    search_ranges = []
    
    print("=== Formal SEARCHKDTREE(ν,R) Algorithm Tests ===")
    for i, test_case in enumerate(test_ranges, 1):
        R = test_case['R']
        description = test_case['description']
        
        print(f"\nTest {i}: {description}")
        print(f"Search range R = {R}")
        
        results = kdtree.range_search(R)
        print("-" * 70)
        
    # Test with different starting nodes (demonstrating ν parameter)
    print("\n=== Testing with Different Starting Nodes (ν) ===")
    
    # Test 1: Search from root
    print("Test A: Starting from root node")
    results_root = kdtree.range_search([[3, 7], [7, 9]])
    
    # Test 2: Search from a specific subtree
    if kdtree.root and kdtree.root.left:
        print(f"\nTest B: Starting from left subtree root: {kdtree.root.left.point}")
        results_left = kdtree.range_search([[3, 7], [7, 9]], start_node=kdtree.root.left)
    
    if kdtree.root and kdtree.root.right:
        print(f"\nTest C: Starting from right subtree root: {kdtree.root.right.point}")
        results_right = kdtree.range_search([[3, 7], [7, 9]], start_node=kdtree.root.right)
    
    # Visualize the tree and search ranges
    try:
        kdtree.visualize(search_ranges)
    except ImportError as e:
        print(f"Visualization requires matplotlib: {e}")
    
    return kdtree

def test_original_case_only():
   

   sample_points = [
       [1, 9], [2, 3], [4, 1], [3, 7], [5, 4], 
       [6, 8], [7, 2], [8, 8], [7, 9], [9, 6]
   ]
   
   print("Sample Points:", sample_points)
   print()
   
   # Build the KD-tree
   kdtree = KDTree()
   kdtree.build(sample_points)
   
   # Print tree structure
   print("=== Tree Structure ===")
   kdtree.print_tree()
   print()
   
   v = [2, 3]
   R = [[3, 7], [7, 9]]
   
   print("=== TESTING ORIGINAL CASE ONLY ===")
   print(f"v = {v}")
   print(f"R = {R}")
   print("Finding points in range R that are below v (y ≤ 3)...")
   
   results = kdtree.range_search(R, query_point=v)
   
   print(f"\nFinal result: {results}")
   kdtree.visualize(query_ranges=[R])

# Run the test
if __name__ == "__main__":
   test_original_case_only()

