from typing import List, Tuple
#from utils import ccw


def ccw(X : list[int], Y: list[int]) -> bool:
    """
    Determine if the points (X[0], Y[0]), (X[1], Y[1]), (X[2], Y[2]) are in counterclockwise order.
    
    Args:
        X: List of x-coordinates of the points.
        Y: List of y-coordinates of the points.
        
    Returns:
        True if the points are in counterclockwise order, False otherwise.
    """
    return (Y[1] - Y[0]) * (X[2] - X[1]) > (Y[2] - Y[1]) * (X[1] - X[0])


class ConvexHull:
    def __init__(self) -> None:
        self.points: List[Tuple[int, int]] = []

    def add_point(self, x: int, y: int) -> None:
        """
        Add a point to the convex hull.
        
        Args:
            x: x-coordinate of the point.
            y: y-coordinate of the point.
        """
        self.points.append((x, y))
    
    def add_points(self, points: List[Tuple[int, int]]) -> None:
        """
        Add multiple points to the convex hull.
        
        Args:
            points: List of tuples where each tuple contains the x and y coordinates of a point.
        """
        self.points.extend(points)
    

    def jarvismarch(self, visualise: bool = False) -> List[Tuple[int, int]]:
        """
        Compute the convex hull using the Jarvis March algorithm.
        
        Args:
            visualise: If True, visualise the convex hull using matplotlib.
        
        Returns:
            A list of points in the convex hull in counterclockwise order.
        """
        if len(self.points) < 3:
            return self.points
        
        hull: List[Tuple[int, int]] = []
        leftmost: Tuple[int, int] = min(self.points, key=lambda p: p[0]) 
        point_on_hull: Tuple[int, int] = leftmost
        
        while True:
            hull.append(point_on_hull) 
            next_point: Tuple[int, int] = self.points[0] 
            
            for p in self.points[1:]:
                if (next_point == point_on_hull or
                    ccw(
                        X=[point_on_hull[0], next_point[0], p[0]],
                        Y=[point_on_hull[1], next_point[1], p[1]]
                    )):
                    next_point = p 
            point_on_hull = next_point
            if point_on_hull == leftmost:
                break

        if visualise:
            import matplotlib.pyplot as plt
            x_coords, y_coords = zip(*self.points)
            hull_x, hull_y = zip(*hull)
            
            plt.figure(figsize=(8, 8))
            plt.scatter(x_coords, y_coords, label="Points")
            plt.plot(hull_x + (hull_x[0],), hull_y + (hull_y[0],), 'r-', label="Convex Hull using JarvisScan")
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.axis('equal') 
            #plt.savefig("convex_hull_jarvis.png")
            plt.show()
        
        return hull
    
    def grahamscan(self, visualise: bool = False) -> List[Tuple[int, int]]:
        """
        Compute the convex hull using the Graham Scan algorithm.

        This corrected version fixes the orientation check and uses a more
        robust method for sorting and building the hull.

        Args:
            visualise: If True, visualise the convex hull using matplotlib.

        Returns:
            A list of points in the convex hull in counterclockwise order.
        """
        if len(self.points) < 3:
            return self.points

        # 1. Find the anchor point (lowest y-coordinate, then leftmost).
        # This is guaranteed to be on the hull.
        lowest_point: Tuple[int, int] = min(self.points, key=lambda p: (p[1], p[0]))

        # 2. Sort the points by polar angle with respect to the lowest_point.
        # If two points have the same angle, the one closer to the anchor comes first.
        def polar_angle_key(p: Tuple[int, int]) -> Tuple[float, float]:
            if p == lowest_point:
                # Ensure the lowest point is treated as the first point
                return (-float('inf'), 0)
            
            dx: int = p[0] - lowest_point[0]
            dy: int = p[1] - lowest_point[1]
            
            # Calculate polar angle and squared distance for efficiency 
            from math import atan2
            angle: float = atan2(dy, dx)
            dist_sq: int = dx*dx + dy*dy
            
            return (angle, dist_sq)

        sorted_points: List[Tuple[int, int]] = sorted(self.points, key=polar_angle_key)
        
        # 3. Build the hull.
        hull: List[Tuple[int, int]] = []

        def orientation(p: Tuple[int, int], q: Tuple[int, int], r: Tuple[int, int]) -> int:
            """
            Calculates the orientation of the ordered triplet (p, q, r).
            This is the cross-product of vectors (q-p) and (r-q).
            Returns:
                 > 0 for a counter-clockwise (left) turn at q.
                 < 0 for a clockwise (right) turn at q.
                 = 0 for collinear points.
            """
            val: int = (q[0] - p[0]) * (r[1] - q[1]) - \
                  (q[1] - p[1]) * (r[0] - q[0])
            return val

        for p in sorted_points:

            while len(hull) >= 2 and orientation(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        

        if visualise:
            import matplotlib.pyplot as plt

            x_coords, y_coords = zip(*self.points)
            

            hull_plot_points = hull + [hull[0]]
            hull_x, hull_y = zip(*hull_plot_points)
            
            plt.figure(figsize=(8, 8))
            plt.scatter(x_coords, y_coords, color='blue', label="All Points")
            plt.plot(hull_x, hull_y, 'r-', label="Convex Hull")
            plt.scatter(hull_x, hull_y, color='red') # Emphasize hull vertices
            
            plt.title("Convex Hull")
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.axis('equal') 
            plt.savefig("convex_hull_graham.png")
            plt.show()
        
        return hull
    


    def divide_and_conquer(self, visualise: bool = False) -> List[Tuple[int, int]]:
        """
        Compute the convex hull using the Divide and Conquer algorithm.
        """
        # The algorithm requires points to be sorted by x-coordinate.
        points = sorted(self.points)
        
        hull = self._divide_and_conquer_recursive(points)
        
        if visualise:
            self._visualise_hull(hull, "Divide and Conquer")
            
        return hull

    def _divide_and_conquer_recursive(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Recursive helper for the divide and conquer algorithm."""
        n = len(points)
        if n <= 5:
            # For small n, use Graham Scan as a base case for efficiency
            # Create a temporary solver for the subset of points
            temp_solver = ConvexHull()
            temp_solver.add_points(points)
            return temp_solver.grahamscan()

        # 1. Divide
        mid = n // 2
        left_hull = self._divide_and_conquer_recursive(points[:mid])
        right_hull = self._divide_and_conquer_recursive(points[mid:])

        # 2. Conquer (Merge)
        return self._merge_hulls(left_hull, right_hull)

    def _merge_hulls(self, left_hull: List[Tuple[int, int]], right_hull: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merges two convex hulls into one."""
        # Find the upper and lower tangents
        upper_l, upper_r = self._find_upper_tangent(left_hull, right_hull)
        lower_l, lower_r = self._find_lower_tangent(left_hull, right_hull)

        # Construct the merged hull
        result = []
        
        # Add points from left hull (from upper tangent to lower tangent, CCW)
        i = upper_l
        while True:
            result.append(left_hull[i])
            if i == lower_l:
                break
            i = (i + 1) % len(left_hull)

        # Add points from right hull (from lower tangent to upper tangent, CCW)
        j = lower_r
        while True:
            result.append(right_hull[j])
            if j == upper_r:
                break
            j = (j + 1) % len(right_hull)
            
        return result

    def _find_upper_tangent(self, left_hull: List[Tuple[int, int]], right_hull: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Finds the upper tangent of two convex hulls."""
        # Start with the rightmost point of the left hull and leftmost of the right
        i = max(range(len(left_hull)), key=lambda idx: left_hull[idx][0])
        j = min(range(len(right_hull)), key=lambda idx: right_hull[idx][0])

        while True:
            prev_i, prev_j = i, j
            # Move j counter-clockwise while the triplet (i, j, next_j) is not a right turn
            while self.orientation(left_hull[i], right_hull[j], right_hull[(j + 1) % len(right_hull)]) >= 0:
                j = (j + 1) % len(right_hull)
            
            # Move i clockwise while the triplet (j, i, next_i) is not a left turn
            while self.orientation(right_hull[j], left_hull[i], left_hull[(i - 1 + len(left_hull)) % len(left_hull)]) <= 0:
                i = (i - 1 + len(left_hull)) % len(left_hull)

            if i == prev_i and j == prev_j: # No change in this iteration
                break
        return i, j

    def _find_lower_tangent(self, left_hull: List[Tuple[int, int]], right_hull: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Finds the lower tangent of two convex hulls."""
        i = max(range(len(left_hull)), key=lambda idx: left_hull[idx][0])
        j = min(range(len(right_hull)), key=lambda idx: right_hull[idx][0])

        while True:
            prev_i, prev_j = i, j
            # Move j clockwise while the triplet (i, j, next_j) is not a left turn
            while self.orientation(left_hull[i], right_hull[j], right_hull[(j - 1 + len(right_hull)) % len(right_hull)]) <= 0:
                j = (j - 1 + len(right_hull)) % len(right_hull)
            
            # Move i counter-clockwise while the triplet (j, i, next_i) is not a right turn
            while self.orientation(right_hull[j], left_hull[i], left_hull[(i + 1) % len(left_hull)]) >= 0:
                i = (i + 1) % len(left_hull)
            
            if i == prev_i and j == prev_j: # No change in this iteration
                break
        return i, j

    @staticmethod
    def orientation(p: Tuple[int, int], q: Tuple[int, int], r: Tuple[int, int]) -> int:
        """
        Calculates the orientation of the ordered triplet (p, q, r).
        This is the cross-product of vectors (q-p) and (r-q).
        Returns:
             > 0 for a counter-clockwise (left) turn at q.
             < 0 for a clockwise (right) turn at q.
             = 0 for collinear points.
        """
        val = (q[0] - p[0]) * (r[1] - q[1]) - \
              (q[1] - p[1]) * (r[0] - q[0])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else -1  # Clockwise or Counterclockwise
    
    def _visualise_hull(self, hull: List[Tuple[int, int]], method_name: str):
        """Helper function to plot the points and the computed hull."""
        import matplotlib.pyplot as plt
        
        x_coords, y_coords = zip(*self.points)
        
        hull_plot_points = hull + [hull[0]]
        hull_x, hull_y = zip(*hull_plot_points)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords, color='blue', label="All Points", zorder=2)
        plt.plot(hull_x, hull_y, 'r-', label="Convex Hull", zorder=3)
        plt.scatter(hull_x[:-1], hull_y[:-1], color='red', s=50, zorder=4) # Emphasize hull vertices
        
        plt.title(f"Convex Hull via {method_name}")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.show()
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage
    ch = ConvexHull()
    #points = [(0, 0), (0, 4), (-4, 0), (5, 0), (0, -6), (1, 0)]
    #points = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 1), (3, 3), (2, 1), (1, 3)]
    points = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (2, -3), (2, 5), (-1, 1)]
    
    ch.add_points(points)

    plt.scatter(*zip(*points), color='blue', label='Input Points')
    plt.title("Input Points")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("input_points.png")
    plt.show()



    
    hull_jarvismarch = ch.jarvismarch(visualise=False)
    hull_grahamscan = ch.grahamscan(visualise=True)
    hull_divide_and_conquer = ch.divide_and_conquer(visualise=True)





 
