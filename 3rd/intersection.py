from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Any
from enum import Enum
from collections import defaultdict
import heapq
from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np


class EventType(Enum):
    """Types of events in the sweep line algorithm."""
    START = "start"
    END = "end"
    INTERSECTION = "intersection"


@dataclass(frozen=True)
class Point:
    """Represents a 2D point with exact arithmetic using Fractions."""
    x: Fraction
    y: Fraction
    
    def __init__(self, x: float | Fraction, y: float | Fraction):
        object.__setattr__(self, 'x', Fraction(x) if not isinstance(x, Fraction) else x)
        object.__setattr__(self, 'y', Fraction(y) if not isinstance(y, Fraction) else y)
    
    def __lt__(self, other: Point) -> bool:
        """Lexicographic ordering for sweep line processing."""
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def to_float_tuple(self) -> Tuple[float, float]:
        """Convert to float tuple for visualization."""
        return (float(self.x), float(self.y))


@dataclass(frozen=True)
class LineSegment:
    """Represents a line segment with two endpoints."""
    start: Point
    end: Point
    id: int = field(default_factory=lambda: LineSegment._next_id())
    
    _id_counter = 0
    
    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def __post_init__(self) -> None:
        """Ensure start point is lexicographically smaller than end point."""
        if self.end < self.start:
            object.__setattr__(self, 'start', self.end)
            object.__setattr__(self, 'end', self.start)
    
    @property
    def upper_endpoint(self) -> Point:
        """Get the upper endpoint (leftmost, or topmost if same x)."""
        return self.start
    
    @property
    def lower_endpoint(self) -> Point:
        """Get the lower endpoint (rightmost, or bottommost if same x)."""
        return self.end
    
    def is_horizontal(self) -> bool:
        """Check if the segment is horizontal."""
        return self.start.y == self.end.y
    
    def is_vertical(self) -> bool:
        """Check if the segment is vertical."""
        return self.start.x == self.end.x
    
    def contains_point(self, point: Point, tolerance: float = 1e-10) -> bool:
        """Check if the segment contains the given point."""
        # Check if point is within the segment bounds first (faster)
        min_x: Fraction = min(self.start.x, self.end.x)
        max_x: Fraction = max(self.start.x, self.end.x)
        min_y: Fraction = min(self.start.y, self.end.y)
        max_y: Fraction = max(self.start.y, self.end.y)

        if not (min_x <= point.x <= max_x and min_y <= point.y <= max_y):
            return False
        
        return self._point_on_line(point)

    def _point_on_line(self, point: Point, tolerance: float = 1e-10) -> bool:
        """Check if point lies on the infinite line containing this segment."""
        if self.is_vertical():
            return abs(point.x - self.start.x) <= tolerance

        if self.is_horizontal():
            return abs(point.y - self.start.y) <= tolerance

        # Use cross product to check collinearity
        dx1: Fraction = point.x - self.start.x
        dy1: Fraction = point.y - self.start.y
        dx2: Fraction = self.end.x - self.start.x
        dy2: Fraction = self.end.y - self.start.y

        return abs(dx1 * dy2 - dy1 * dx2) <= tolerance

    def y_at_x(self, x: Fraction) -> Fraction:
        """Get y-coordinate at given x-coordinate on the segment."""
        if self.is_vertical():
            if x != self.start.x:
                raise ValueError("x not on vertical segment")
            return self.start.y
        
        if self.start.x == self.end.x:
            return self.start.y
        
        # Linear interpolation
        t: Fraction = (x - self.start.x) / (self.end.x - self.start.x)
        return self.start.y + t * (self.end.y - self.start.y)
    
    def intersect(self, other: LineSegment) -> Optional[Point]:
        """Find intersection point with another line segment."""
        # Get line parameters
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        x3, y3 = other.start.x, other.start.y
        x4, y4 = other.end.x, other.end.y
        
        # Calculate denominators for parametric equations
        denom: Fraction = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if denom == 0:
            # Parallel or collinear
            return None
        
        # Calculate intersection parameters
        t: Fraction = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u: Fraction = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            ix: Fraction = x1 + t * (x2 - x1)
            iy: Fraction = y1 + t * (y2 - y1)
            return Point(ix, iy)
        
        return None
    
    def __str__(self) -> str:
        return f"Segment{self.id}({self.start} -> {self.end})"


@dataclass
class Event:
    """Represents an event in the sweep line algorithm."""
    point: Point
    event_type: EventType
    segments: Set[LineSegment] = field(default_factory=set)
    
    def __lt__(self, other: Event) -> bool:
        """Events are ordered by point, then by type priority."""
        if self.point != other.point:
            return self.point < other.point
        
        # Priority: START < INTERSECTION < END
        priority: Dict[EventType, int] = {EventType.START: 0, EventType.INTERSECTION: 1, EventType.END: 2}
        return priority[self.event_type] < priority[other.event_type]


class StatusStructure:
    """
    Manages the sweep line status - segments intersected by current sweep line.
    Maintains segments in order of intersection with sweep line.
    """
    
    def __init__(self) -> None:
        self.segments: List[LineSegment] = []
        self.current_x: Optional[Fraction] = None
    
    def set_sweep_line_x(self, x: Fraction) -> None:
        """Update the current sweep line position."""
        self.current_x = x
        self._sort_segments()

    def _sort_segments(self) -> None:
        """Sort segments by y-coordinate at current sweep line."""
        if self.current_x is None:
            return
        
        def segment_y_at_sweep_line(seg: LineSegment) -> Tuple[Fraction, bool, int]:
            """Get sorting key for segment at current sweep line."""
            if seg.is_vertical():
                # For vertical segments at the sweep line, use midpoint y
                avg_y = (seg.start.y + seg.end.y) / 2
                return (avg_y, True, seg.id)  # True indicates vertical
            
            # Check if the current x is within the segment's x range
            min_x = min(seg.start.x, seg.end.x)
            max_x = max(seg.start.x, seg.end.x)
            
            if min_x <= self.current_x <= max_x: # type: ignore
                try:
                    y = seg.y_at_x(self.current_x) # type: ignore
                    return (y, seg.is_horizontal(), seg.id)
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Fallback: use midpoint if segment doesn't intersect sweep line properly
            avg_y = (seg.start.y + seg.end.y) / 2
            return (avg_y, seg.is_horizontal(), seg.id)
        
        self.segments.sort(key=segment_y_at_sweep_line)
    
    def insert(self, segment: LineSegment) -> None:
        """Insert a segment into the status structure."""
        if segment not in self.segments:
            self.segments.append(segment)
            self._sort_segments()
    
    def remove(self, segment: LineSegment) -> None:
        """Remove a segment from the status structure."""
        if segment in self.segments:
            self.segments.remove(segment)
    
    def find_containing_segments(self, point: Point) -> List[LineSegment]:
        """Find all segments that contain the given point."""
        containing = []
        for seg in self.segments:
            if seg.contains_point(point):
                containing.append(seg)
        return containing
    
    def get_neighbors(self, segment: LineSegment) -> Tuple[Optional[LineSegment], Optional[LineSegment]]:
        """Get left and right neighbors of a segment."""
        try:
            idx: int = self.segments.index(segment)
            left: LineSegment | None = self.segments[idx - 1] if idx > 0 else None
            right: LineSegment | None = self.segments[idx + 1] if idx < len(self.segments) - 1 else None
            return left, right
        except ValueError:
            return None, None
    
    def get_leftmost_rightmost(self, segments: Set[LineSegment]) -> Tuple[Optional[LineSegment], Optional[LineSegment]]:
        """Get leftmost and rightmost segments from a set."""
        if not segments:
            return None, None

        segment_indices: List[Tuple[int, LineSegment]] = []
        for seg in segments:
            try:
                segment_indices.append((self.segments.index(seg), seg))
            except ValueError:
                continue
        
        if not segment_indices:
            return None, None
        
        segment_indices.sort()
        return segment_indices[0][1], segment_indices[-1][1]


class SweepLineIntersection:
    """
    Implementation of the Sweep Line Algorithm
    for detecting line segment intersections.
    """
    
    def __init__(self) -> None:
        self.event_queue: List[Event] = []
        self.status: StatusStructure = StatusStructure()
        self.intersections: List[Tuple[Point, Set[LineSegment]]] = []
        self.processed_intersections: Set[Point] = set()
    
    def find_intersections(self, segments: List[LineSegment]) -> List[Tuple[Point, Set[LineSegment]]]:
        """
        Main algorithm to find all intersection points among line segments.
        
        Args:
            segments: List of line segments to process
            
        Returns:
            List of tuples containing intersection points and the segments that intersect there
        """
        self._initialize(segments)
        
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self._handle_event_point(event)
        
        return self.intersections
    
    def _initialize(self, segments: List[LineSegment]):
        """Initialize the event queue with merged endpoint events and mid-segment intersections."""
        self.event_queue = []
        self.status = StatusStructure()
        self.intersections = []
        self.processed_intersections = set()

        # Map point -> (segments_starting, segments_ending)
        point_map: Dict[Point, Tuple[Set[LineSegment], Set[LineSegment]]] = {}
        for segment in segments:
            start_pt: Point = segment.upper_endpoint
            end_pt: Point = segment.lower_endpoint

            if start_pt not in point_map:
                point_map[start_pt] = (set(), set())
            if end_pt not in point_map:
                point_map[end_pt] = (set(), set())

            point_map[start_pt][0].add(segment)  # segments starting here
            point_map[end_pt][1].add(segment)    # segments ending here

        # Create events for endpoints (merge starts/ends at same point)
        for pt, (starts, ends) in point_map.items():
            combined_segments = starts | ends
            if starts and ends:
                ev_type = EventType.INTERSECTION
            elif starts:
                ev_type = EventType.START
            else:
                ev_type = EventType.END
            ev = Event(point=pt, event_type=ev_type, segments=combined_segments)
            heapq.heappush(self.event_queue, ev)

        # Detect mid-segment intersections (not endpoints for both)
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                inter_pt = seg1.intersect(seg2)
                if inter_pt:
                    # Skip if intersection is exactly a shared endpoint already processed
                    if inter_pt in point_map:
                        continue
                    # Ensure point lies strictly inside at least one of the segments
                    if ((seg1.upper_endpoint != inter_pt and seg1.lower_endpoint != inter_pt) or
                        (seg2.upper_endpoint != inter_pt and seg2.lower_endpoint != inter_pt)):
                        ev = Event(point=inter_pt,
                                event_type=EventType.INTERSECTION,
                                segments={seg1, seg2})
                        heapq.heappush(self.event_queue, ev)

    
    def _handle_event_point(self, event: Event):
        """Handle an event point according to the sweep line algorithm."""
        p: Point = event.point
        self.status.set_sweep_line_x(p.x)
        
        # Find segments with upper endpoint at p
        U_p: set[LineSegment] = set()
        if event.event_type == EventType.START:
            U_p = event.segments.copy()
        
        # Find segments stored in T that contain p
        containing_segments: List[LineSegment] = self.status.find_containing_segments(p)
        
        # Separate into L(p) and C(p)
        L_p: set[LineSegment] = set()  # Segments whose lower endpoint is p
        C_p: set[LineSegment] = set()  # Segments that contain p in their interior

        for seg in containing_segments:
            if seg.lower_endpoint == p:
                L_p.add(seg)
            elif seg.upper_endpoint != p and seg not in U_p:  # Interior point, not starting
                C_p.add(seg)
        
        # For intersection events, add the intersecting segments to C(p)
        if event.event_type == EventType.INTERSECTION:
            # These segments intersect at p but don't start or end there
            for seg in event.segments:
                if seg.upper_endpoint != p and seg.lower_endpoint != p:
                    C_p.add(seg)
        
        # Check if this is an intersection (more than one segment involved)
        all_segments: Set[LineSegment] = L_p | U_p | C_p
        if len(all_segments) > 1:
            if p not in self.processed_intersections:
                self.intersections.append((p, all_segments))
                self.processed_intersections.add(p)
        
        # CRITICAL: Get adjacent segments BEFORE removal to check for new events
        segments_to_remove: Set[LineSegment] = L_p | C_p
        adjacent_pairs_to_check: List[Tuple[LineSegment, LineSegment]] = []

        # For each segment being removed, find its neighbors that will become adjacent
        for seg in segments_to_remove:
            left_neighbor, right_neighbor = self.status.get_neighbors(seg)
            if left_neighbor and right_neighbor:
                # These two segments will become adjacent after seg is removed
                adjacent_pairs_to_check.append((left_neighbor, right_neighbor))
        
        # Remove segments in L(p) ∪ C(p) from T
        for seg in segments_to_remove:
            self.status.remove(seg)
        
        # Insert segments in U(p) ∪ C(p) into T
        segments_to_insert = U_p | C_p
        for seg in segments_to_insert:
            self.status.insert(seg)
        
        # CRITICAL: Check newly adjacent pairs for intersections
        for left_seg, right_seg in adjacent_pairs_to_check:
            if left_seg not in segments_to_remove and right_seg not in segments_to_remove:
                self._find_new_event(left_seg, right_seg, p)
        
        # Find new events with neighbors of inserted segments
        if segments_to_insert:
            # Get leftmost and rightmost segments that were inserted
            leftmost, rightmost = self.status.get_leftmost_rightmost(segments_to_insert)
            
            if leftmost:
                left_neighbor, _ = self.status.get_neighbors(leftmost)
                if left_neighbor and left_neighbor not in segments_to_insert:
                    self._find_new_event(left_neighbor, leftmost, p)
            
            if rightmost:
                _, right_neighbor = self.status.get_neighbors(rightmost)
                if right_neighbor and right_neighbor not in segments_to_insert:
                    self._find_new_event(rightmost, right_neighbor, p)
    
    def _find_new_event(self, seg1: LineSegment, seg2: LineSegment, current_point: Point) -> None:
        """Find and add intersection events between two segments."""
        if not seg1 or not seg2 or seg1 == seg2:
            return
        
        intersection = seg1.intersect(seg2)
        if not intersection:
            return
        
        # Check if intersection is to the right of current sweep line, or below on same x
        is_future_event = (intersection.x > current_point.x or 
                          (abs(intersection.x - current_point.x) < 1e-10 and intersection.y >= current_point.y))

        if is_future_event and intersection not in self.processed_intersections:
            # Check if this exact intersection event already exists in queue
            event_exists = False
            for e in self.event_queue:
                if (e.point == intersection and 
                    e.event_type == EventType.INTERSECTION):
                    # Add segments to existing event if not already there
                    e.segments.add(seg1)
                    e.segments.add(seg2)
                    event_exists = True
                    break
            
            if not event_exists:
                new_event = Event(
                    point=intersection,
                    event_type=EventType.INTERSECTION,
                    segments={seg1, seg2}
                )
                heapq.heappush(self.event_queue, new_event)


def visualize_segments_and_intersections(
    segments: List[LineSegment], 
    intersections: List[Tuple[Point, Set[LineSegment]]],
    title: str = "Line Segment Intersections",
    save: bool = False
) -> None:
    """Visualize line segments and their intersection points."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot segments
    cmap = plt.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        x_coords = [float(segment.start.x), float(segment.end.x)]
        y_coords = [float(segment.start.y), float(segment.end.y)]
        
        ax.plot(x_coords, y_coords, 'o-', 
                color=colors[i], linewidth=2, markersize=6,
                label=f'Segment {segment.id}')
    
    # Plot intersections
    if intersections:
        intersection_points = [point for point, _ in intersections]
        x_intersections = [float(p.x) for p in intersection_points]
        y_intersections = [float(p.y) for p in intersection_points]
        
        ax.scatter(x_intersections, y_intersections, 
                  color='red', s=100, zorder=5, marker='*',
                  label=f'Intersections ({len(intersections)})')
        
        # Annotate intersection points
        for i, (point, segments_at_point) in enumerate(intersections):
            ax.annotate(
                f'I{i+1}\n({float(point.x):.2f}, {float(point.y):.2f})',
                (float(point.x), float(point.y)),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=8, ha='left'
            )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{title}.png")
    plt.show()


def create_test_segments(test_case: int) -> List[LineSegment]:
    """Create test cases for the algorithm."""
    if test_case == 1:
        # Original test case from assignment
        return [
            LineSegment(Point(1, 5), Point(4, 5)),    # Horizontal
            LineSegment(Point(2, 5), Point(10, 1)),   # Diagonal
            LineSegment(Point(3, 2), Point(10, 3)),   # Diagonal
            LineSegment(Point(6, 4), Point(9, 4)),    # Horizontal
            LineSegment(Point(7, 1), Point(8, 1)),    # Horizontal
        ]
    elif test_case == 2:
        # Cross pattern
        return [
            LineSegment(Point(0, 0), Point(4, 4)),    # Diagonal /
            LineSegment(Point(0, 4), Point(4, 0)),    # Diagonal \
            LineSegment(Point(2, 0), Point(2, 4)),    # Vertical |
            LineSegment(Point(0, 2), Point(4, 2)),    # Horizontal -
        ]
    elif test_case == 3:
        # Triangle with internal segments
        return [
            LineSegment(Point(0, 0), Point(6, 0)),    # Base
            LineSegment(Point(0, 0), Point(3, 4)),    # Left side
            LineSegment(Point(6, 0), Point(3, 4)),    # Right side
            LineSegment(Point(1, 1), Point(5, 1)),    # Internal horizontal
            LineSegment(Point(2, 0), Point(2, 3)),    # Internal vertical
        ]
    else:
        raise ValueError(f"Unknown test case: {test_case}")


def run_test_case(test_case: int, save: bool = False) -> None:
    """Run a specific test case and display results."""
    print(f"\n{'='*50}")
    print(f"TEST CASE {test_case}")
    print(f"{'='*50}")
    
    segments = create_test_segments(test_case)
    
    # Display input segments
    print("\nInput Line Segments:")
    for segment in segments:
        start_float = segment.start.to_float_tuple()
        end_float = segment.end.to_float_tuple()
        print(f"  {segment}: {start_float} -> {end_float}")
    
    # Run algorithm
    sweep_line = SweepLineIntersection()
    intersections = sweep_line.find_intersections(segments)
    
    # Display results
    print(f"\nFound {len(intersections)} intersection point(s):")
    
    for i, (point, segments_at_point) in enumerate(intersections):
        point_float = point.to_float_tuple()
        segment_ids = [seg.id for seg in segments_at_point]
        print(f"  Intersection {i+1}")
        print(f"    Exact coordinates: ({point.x: .2f}, {point.y: .2f})")
        print(f"    Segments involved: {sorted(segment_ids)}")
    
    # Visualize
    visualize_segments_and_intersections(
        segments, intersections, 
        title=f"Test Case {test_case} - Line Segment Intersections",
        save=save
    )


def main() -> None:
    """Run all test cases."""
    print("Sweep Line Algorithm for Line Segment Intersections")
    
    # Run all test cases
    for test_case in range(1, 4):
        try:
            run_test_case(test_case, save=True)
        except Exception as e:
            print(f"Error in test case {test_case}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("All test cases completed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
