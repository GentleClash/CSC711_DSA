import math
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Point:
    """Represents a 2D point with x and y coordinates."""
    x: float
    y: float
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

class ClosestPairFinder:
    """Divide-and-conquer algorithm to find closest pair of points."""
    
    @staticmethod
    def distance(p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    @staticmethod
    def brute_force_closest_pair(points: List[Point]) -> Tuple[Point, Point, float]:
        """
        Brute force method for small sets of points (≤ 3 points).
        Time complexity: O(n²)
        """
        n = len(points)
        if n < 2:
            raise ValueError("Need at least 2 points")
        if n == 2:
            return points[0], points[1], ClosestPairFinder.distance(points[0], points[1])
        
        min_dist = float('inf')
        closest_pair: Tuple[Point, Point] = (points[0], points[1])
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = ClosestPairFinder.distance(points[i], points[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (points[i], points[j])
        
        return closest_pair[0], closest_pair[1], min_dist
    
    @staticmethod
    def closest_pair_in_strip(strip: List[Point], delta: float) -> Tuple[Point, Point, float]:
        """
        Find closest pair in the vertical strip around the dividing line.
        Points in strip are sorted by y-coordinate.
        
        Args:
            strip: Points in the strip, sorted by y-coordinate
            delta: Current minimum distance found
            
        Returns:
            Closest pair and distance in the strip
        """
        min_dist: float = delta
        closest_pair: Tuple[Point, Point] = (None, None)

        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j].y - strip[i].y) < min_dist:
                dist = ClosestPairFinder.distance(strip[i], strip[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (strip[i], strip[j])
                j += 1
        
        return closest_pair[0], closest_pair[1], min_dist
    
    @staticmethod
    def closest_pair_rec(px: List[Point], py: List[Point]) -> Tuple[Point, Point, float]:
        """
        Recursive divide-and-conquer algorithm to find closest pair.
        
        Args:
            px: Points sorted by x-coordinate
            py: Points sorted by y-coordinate
            
        Returns:
            Tuple of (point1, point2, minimum_distance)
        """
        n = len(px)
        
        # Base case: use brute force for small sets
        if n <= 3:
            return ClosestPairFinder.brute_force_closest_pair(px)
        
        # Divide: find the middle point
        mid = n // 2
        midpoint = px[mid]
        
        # Split points by x-coordinate
        pxl: List[Point] = px[:mid]  # Left half
        pxr: List[Point] = px[mid:]  # Right half
        
        # Split points by y-coordinate while maintaining the division
        pyl: List[Point] = [point for point in py if point.x <= midpoint.x]
        pyr: List[Point] = [point for point in py if point.x > midpoint.x]
        
        # Conquer: recursively find closest pairs in both halves
        p1_left, p2_left, dl = ClosestPairFinder.closest_pair_rec(pxl, pyl)
        p1_right, p2_right, dr = ClosestPairFinder.closest_pair_rec(pxr, pyr)
        
        # Find the minimum of the two halves
        if dl <= dr:
            delta = dl
            closest_pair: Tuple[Point, Point] = (p1_left, p2_left)
        else:
            delta = dr
            closest_pair: Tuple[Point, Point] = (p1_right, p2_right)
        
        # Create strip of points close to the dividing line
        strip: List[Point] = [point for point in py if abs(point.x - midpoint.x) < delta]
        
        # Find closest pair in the strip
        p1_strip, p2_strip, ds = ClosestPairFinder.closest_pair_in_strip(strip, delta)
        
        # Return the overall closest pair
        if ds < delta:
            return p1_strip, p2_strip, ds
        else:
            return closest_pair[0], closest_pair[1], delta
    
    @staticmethod
    def visualize_points(points: List[Point], point1: Point, point2: Point) -> None:
        """Visualize the points and the closest pair."""
        plt.figure(figsize=(8, 6))
        plt.scatter([p.x for p in points], [p.y for p in points], color='blue', label='Points')
        plt.scatter([point1.x, point2.x], [point1.y, point2.y], color='red', label='Closest Pair', s=100)
        plt.plot([point1.x, point2.x], [point1.y, point2.y], color='red', linestyle='--')
        plt.title("Closest Pair of Points")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()
    
    @staticmethod
    def find_closest_pair(points: List[Point], visualize: bool = False) -> Tuple[Point, Point, float]:
        """
        Main function to find the closest pair of points.
        
        Args:
            points: List of 2D points
            
        Returns:
            Tuple of (point1, point2, minimum_distance)
            
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points to find closest pair")
        
        # Preprocess
        px = sorted(points, key=lambda p: p.x)  
        py = sorted(points, key=lambda p: p.y)  
        
        # Call recursive function
        point1, points2, min_distance = ClosestPairFinder.closest_pair_rec(px, py)

        if visualize:
            ClosestPairFinder.visualize_points(points, point1, points2)

        return point1, points2, min_distance


def main():
    """Example usage and testing."""
    
    # Test case 1: Simple case
    points1 = [
        Point(2, 3),
        Point(12, 30),
        Point(40, 50),
        Point(5, 1),
        Point(12, 10),
        Point(3, 4)
    ]
    
    print("Test Case 1:")
    print(f"Points: {[str(p) for p in points1]}")
    p1, p2, dist = ClosestPairFinder.find_closest_pair(points1, visualize=True)
    print(f"Closest pair: {p1} and {p2}")
    print(f"Distance: {dist:.4f}")
    print()
    
    # Test case 2
    points2 = [
        Point(1, 1),
        Point(2, 1),
        Point(3, 1),
        Point(4, 1),
        Point(10, 1)
    ]
    
    print("Test Case 2 (Points on a line):")
    print(f"Points: {[str(p) for p in points2]}")
    p1, p2, dist = ClosestPairFinder.find_closest_pair(points2, visualize=True)
    print(f"Closest pair: {p1} and {p2}")
    print(f"Distance: {dist:.4f}")
    print()
    
    # Test case 3
    points3 = [
        Point(0, 0),
        Point(1, 1),
        Point(2, 2),
        Point(3, 3),
        Point(0.5, 0.5),
        Point(1.5, 1.5),
        Point(10, 10),
        Point(11, 11)
    ]
    
    print("Test Case 3:")
    print(f"Points: {[str(p) for p in points3]}")
    p1, p2, dist = ClosestPairFinder.find_closest_pair(points3, visualize=True)
    print(f"Closest pair: {p1} and {p2}")
    print(f"Distance: {dist:.4f}")


if __name__ == "__main__":
    main()
