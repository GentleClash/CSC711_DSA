import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)


@dataclass
class Edge:
    """Represents an edge in the Voronoi diagram."""
    start: Optional[Point]
    end: Optional[Point]
    site1: Point
    site2: Point

    def __repr__(self) -> str:
        return f"Edge({self.start} -> {self.end}, sites: {self.site1}, {self.site2})"


class VoronoiDiagram:

    def __init__(self, sites: List[Point], bounds: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Initialize Voronoi diagram with site points.

        Args:
            sites: List of site points
            bounds: Bounding box (min_x, max_x, min_y, max_y) for clipping
        """
        self.sites = sites
        self.edges = []
        self.vertices = []

        # Set default bounds if not provided
        if bounds is None:
            if sites:
                xs = [p.x for p in sites]
                ys = [p.y for p in sites]
                margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.5
                self.bounds = (
                    min(xs) - margin, max(xs) + margin,
                    min(ys) - margin, max(ys) + margin
                )
            else:
                self.bounds = (-10, 10, -10, 10)
        else:
            self.bounds = bounds

        self._construct_diagram()


    def _circumcenter(self, p1: Point, p2: Point, p3: Point) -> Optional[Point]:
        """
        Calculate circumcenter of three points.
        Returns None if points are collinear.
        """
        # Use determinant method to find circumcenter
        d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

        if abs(d) < 1e-10:  # Points are collinear
            return None

        ux = ((p1.x**2 + p1.y**2) * (p2.y - p3.y) +
                    (p2.x**2 + p2.y**2) * (p3.y - p1.y) +
                    (p3.x**2 + p3.y**2) * (p1.y - p2.y)) / d

        uy = ((p1.x**2 + p1.y**2) * (p3.x - p2.x) +
                    (p2.x**2 + p2.y**2) * (p1.x - p3.x) +
                    (p3.x**2 + p3.y**2) * (p2.x - p1.x)) / d

        return Point(ux, uy)

    def _construct_diagram(self) -> None:
        """
        Construct the Voronoi diagram.
        """
        n = len(self.sites)
        if n < 2:
            return

        # Find all Voronoi vertices first using Delaunay triangulation approach
        self._find_voronoi_vertices()

        # Create edges by connecting vertices and extending to boundaries
        self._create_edges()

        # Clip edges to bounds
        self._clip_edges()

    def _find_voronoi_vertices(self) -> None:
        """
        Find all Voronoi vertices by checking all combinations of 3 sites.
        A Voronoi vertex is the circumcenter of 3 sites that forms an empty circle.
        """
        n = len(self.sites)
        self.vertices = []

        # Check all combinations of 3 sites
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    site1, site2, site3 = self.sites[i], self.sites[j], self.sites[k]

                    # Calculate circumcenter
                    circumcenter = self._circumcenter(site1, site2, site3)
                    if circumcenter is None:
                        continue

                    # Check if this is a valid Voronoi vertex (empty circle test)
                    circumradius = circumcenter.distance_to(site1)
                    is_valid = True

                    for site in self.sites:
                        if site not in [site1, site2, site3]:
                            if circumcenter.distance_to(site) < circumradius - 1e-10:
                                is_valid = False
                                break

                    if is_valid:
                        # Check if vertex is within reasonable bounds
                        min_x, max_x, min_y, max_y = self.bounds
                        margin = max(max_x - min_x, max_y - min_y) * 2
                        if (min_x - margin <= circumcenter.x <= max_x + margin and
                            min_y - margin <= circumcenter.y <= max_y + margin):
                            self.vertices.append(circumcenter)

    def _create_edges(self) -> None:
        """
        Create Voronoi edges.
        """
        self.edges = []
        n = len(self.sites)

        # For each pair of sites, find their Voronoi edge
        for i in range(n):
            for j in range(i + 1, n):
                site1, site2 = self.sites[i], self.sites[j]

                # Find vertices that lie on the perpendicular bisector of these two sites
                edge_vertices = []
                for vertex in self.vertices:
                    dist1 = vertex.distance_to(site1)
                    dist2 = vertex.distance_to(site2)

                    # Check if vertex is equidistant from both sites
                    if abs(dist1 - dist2) < 1e-8:
                        edge_vertices.append(vertex)

                # Create edge based on number of vertices found
                if len(edge_vertices) == 2:
                    # Finite edge
                    self.edges.append(Edge(edge_vertices[0], edge_vertices[1], site1, site2))

                elif len(edge_vertices) == 1:
                    # Semi-infinite edge
                    self._create_semi_infinite_edge(edge_vertices[0], site1, site2)

                elif len(edge_vertices) == 0:
                    # Completely infinite edge
                    self._create_infinite_edge(site1, site2)

    def _create_semi_infinite_edge(self, vertex: Point, site1: Point, site2: Point) -> None:
        """
        Create a semi-infinite edge starting from a vertex.
        """
        # Direction perpendicular to the line connecting the sites
        dx = site2.x - site1.x
        dy = site2.y - site1.y

        # Perpendicular directions
        perp_dx1, perp_dy1 = -dy, dx
        perp_dx2, perp_dy2 = dy, -dx

        # Normalize
        length = math.sqrt(dx**2 + dy**2)
        if length > 1e-10:
            perp_dx1 /= length
            perp_dy1 /= length
            perp_dx2 /= length
            perp_dy2 /= length

        # Test both directions to find the correct one
        test_distance = min(abs(self.bounds[1] - self.bounds[0]), abs(self.bounds[3] - self.bounds[2])) / 10

        for perp_dx, perp_dy in [(perp_dx1, perp_dy1), (perp_dx2, perp_dy2)]:
            test_point = Point(vertex.x + test_distance * perp_dx, vertex.y + test_distance * perp_dy)

            # Check if moving in this direction takes us away from the Voronoi region
            # The point should remain equidistant from site1 and site2, and not get closer to any other site
            dist1 = test_point.distance_to(site1)
            dist2 = test_point.distance_to(site2)

            if abs(dist1 - dist2) > 1e-6:  # Not on bisector anymore, skip this direction
                continue

            # Check if any other site is closer
            is_valid_direction = True
            for site in self.sites:
                if site != site1 and site != site2:
                    if test_point.distance_to(site) < dist1 - 1e-8:
                        is_valid_direction = False
                        break

            if is_valid_direction:
                # This is the correct direction, extend to boundary
                boundary_point = self._extend_ray_to_boundary(vertex, perp_dx, perp_dy)
                if boundary_point:
                    self.edges.append(Edge(vertex, boundary_point, site1, site2))
                break

    def _create_infinite_edge(self, site1: Point, site2: Point) -> None:
        """
        Create a completely infinite edge (line segment across the bounds).
        """
        # Check if this edge should exist by testing the midpoint
        mid_x = (site1.x + site2.x) / 2
        mid_y = (site1.y + site2.y) / 2
        midpoint = Point(mid_x, mid_y)

        # Check if midpoint is in the Voronoi region for these two sites
        base_dist = midpoint.distance_to(site1)

        is_valid_region = True
        for site in self.sites:
            if site != site1 and site != site2:
                if midpoint.distance_to(site) < base_dist - 1e-8:
                    is_valid_region = False
                    break

        if not is_valid_region:
            return

        # Direction perpendicular to the line connecting the sites
        dx = site2.x - site1.x
        dy = site2.y - site1.y

        # Perpendicular direction
        perp_dx, perp_dy = -dy, dx

        # Normalize
        length = math.sqrt(perp_dx**2 + perp_dy**2)
        if length > 1e-10:
            perp_dx /= length
            perp_dy /= length

        # Extend in both directions from midpoint
        boundary1 = self._extend_ray_to_boundary(midpoint, perp_dx, perp_dy)
        boundary2 = self._extend_ray_to_boundary(midpoint, -perp_dx, -perp_dy)

        if boundary1 and boundary2:
            self.edges.append(Edge(boundary1, boundary2, site1, site2))

    def _extend_ray_to_boundary(self, start: Point, dx: float, dy: float) -> Optional[Point]:
        """
        Extend a ray from start point in direction (dx, dy) to boundary.
        """
        min_x, max_x, min_y, max_y = self.bounds

        # Find intersection with each boundary
        intersections = []

        # Left boundary (x = min_x)
        if abs(dx) > 1e-10:
            t = (min_x - start.x) / dx
            if t > 1e-10:  # Forward direction
                y = start.y + t * dy
                if min_y <= y <= max_y:
                    intersections.append((t, Point(min_x, y)))

        # Right boundary (x = max_x)
        if abs(dx) > 1e-10:
            t = (max_x - start.x) / dx
            if t > 1e-10:  # Forward direction
                y = start.y + t * dy
                if min_y <= y <= max_y:
                    intersections.append((t, Point(max_x, y)))

        # Bottom boundary (y = min_y)
        if abs(dy) > 1e-10:
            t = (min_y - start.y) / dy
            if t > 1e-10:  # Forward direction
                x = start.x + t * dx
                if min_x <= x <= max_x:
                    intersections.append((t, Point(x, min_y)))

        # Top boundary (y = max_y)
        if abs(dy) > 1e-10:
            t = (max_y - start.y) / dy
            if t > 1e-10:  # Forward direction
                x = start.x + t * dx
                if min_x <= x <= max_x:
                    intersections.append((t, Point(x, max_y)))

        # Return the closest intersection
        if intersections:
            intersections.sort(key=lambda x: x[0])
            return intersections[0][1]

        return None


    def _clip_edges(self) -> None:
        """
        Clip edges to the bounding rectangle.
        """
        min_x, max_x, min_y, max_y = self.bounds
        clipped_edges = []

        for edge in self.edges:
            if edge.start is None or edge.end is None:
                continue

            # Simple clipping - just remove edges completely outside bounds
            if (edge.start.x < min_x and edge.end.x < min_x) or \
               (edge.start.x > max_x and edge.end.x > max_x) or \
               (edge.start.y < min_y and edge.end.y < min_y) or \
               (edge.start.y > max_y and edge.end.y > max_y):
                continue

            # Clip to bounds
            start_x = max(min_x, min(max_x, edge.start.x))
            start_y = max(min_y, min(max_y, edge.start.y))
            end_x = max(min_x, min(max_x, edge.end.x))
            end_y = max(min_y, min(max_y, edge.end.y))

            clipped_start = Point(start_x, start_y)
            clipped_end = Point(end_x, end_y)

            if clipped_start.distance_to(clipped_end) > 1e-10:
                clipped_edges.append(Edge(clipped_start, clipped_end, edge.site1, edge.site2))

        self.edges = clipped_edges


def display_voronoi_diagram(voronoi: VoronoiDiagram, title: str = "Voronoi Diagram", filename: Optional[str] = None) -> None:
    """
    Display the Voronoi diagram using matplotlib.

    Args:
        voronoi: VoronoiDiagram instance
        title: Title for the plot
        filename: If provided, save the plot to this file instead of showing it
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot sites
    site_x = [site.x for site in voronoi.sites]
    site_y = [site.y for site in voronoi.sites]
    ax.plot(site_x, site_y, 'ro', markersize=8, label='Sites', zorder=3)

    # Add site labels
    for i, site in enumerate(voronoi.sites):
        ax.annotate(f'P{i+1}({site.x:.1f}, {site.y:.1f})',
                   (site.x, site.y),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   zorder=4)

    # Plot Voronoi edges
    for edge in voronoi.edges:
        if edge.start and edge.end:
            ax.plot([edge.start.x, edge.end.x],
                   [edge.start.y, edge.end.y],
                   'b-', linewidth=1.5, alpha=0.7, zorder=1)

    # Plot Voronoi vertices
    if voronoi.vertices:
        vertex_x = [v.x for v in voronoi.vertices]
        vertex_y = [v.y for v in voronoi.vertices]
        ax.plot(vertex_x, vertex_y, 'go', markersize=4, label='Vertices', zorder=2)

    # Set bounds and styling
    min_x, max_x, min_y, max_y = voronoi.bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved diagram to {filename}")
    else:
        plt.show()


def test_voronoi_diagram() -> VoronoiDiagram:
    """
    Test the Voronoi diagram implementation with the provided sample points.
    """
    # Sample points as specified
    sites = [
        Point(2, 5),  # P1
        Point(4, 5),  # P2
        Point(7, 2),  # P3
        Point(5, 7)   # P4
    ]

    print("Testing Voronoi Diagram Implementation")
    print("=" * 50)
    print(f"Site points: {sites}")

    # Create Voronoi diagram
    voronoi = VoronoiDiagram(sites)

    print(f"\nNumber of sites: {len(voronoi.sites)}")
    print(f"Number of vertices: {len(voronoi.vertices)}")
    print(f"Number of edges: {len(voronoi.edges)}")
    print(f"Bounds: {voronoi.bounds}")

    # Display detailed results
    print("\nVoronoi Vertices:")
    for i, vertex in enumerate(voronoi.vertices):
        print(f"  V{i+1}: {vertex}")

    print("\nVoronoi Edges:")
    for i, edge in enumerate(voronoi.edges):
        print(f"  E{i+1}: {edge}")

    # Display and save the diagram
    display_voronoi_diagram(voronoi, "Voronoi Diagram - Test Points", filename="voronoi_sample_points.png")

    return voronoi


def run_additional_tests() -> None:
    """
    Run additional tests with different point configurations.
    """
    print("\n" + "=" * 50)
    print("Additional Test Cases")
    print("=" * 50)

    # Test case 2: Random points
    print("\nTest 2: Random points")
    import random
    random.seed(42)  # For reproducibility
    random_sites = [Point(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(6)]
    voronoi3 = VoronoiDiagram(random_sites)
    display_voronoi_diagram(voronoi3, "Voronoi Diagram - Random Points", filename="voronoi_random_points.png")


if __name__ == "__main__":
    # Run the main test with provided points
    test_diagram = test_voronoi_diagram()

    # Run additional tests
    run_additional_tests()

    print("\nVoronoi Diagram implementation completed successfully!")
