# Sweep Line Algorithm for Line Segment Intersections

A comprehensive Python implementation of the **Bentley-Ottmann Sweep Line Algorithm** for efficiently detecting all intersection points among a collection of line segments in 2D space.

## Overview

This implementation solves the **Line Segment Intersection Problem** using a sweep line algorithm that runs in O((n + k) log n) time complexity, where:
- `n` = number of line segments
- `k` = number of intersection points

## Algorithm Description

### Pseudo-Algorithm

```
SWEEP_LINE_INTERSECTION(S):
    Input: Set S of n line segments
    Output: All intersection points with involved segments

    1. INITIALIZE:
       - Event queue Q (priority queue)
       - Status structure T (ordered segments crossing sweep line)
       - Result list intersections

    2. PREPROCESS:
       For each segment s in S:
           Add START event at s.upper_endpoint to Q
           Add END event at s.lower_endpoint to Q
       
       For each pair (s1, s2) in S:
           intersection = s1.intersect(s2)
           If intersection exists and is interior to at least one segment:
               Add INTERSECTION event at intersection to Q

    3. SWEEP:
       While Q is not empty:
           event = Q.extract_min()
           HANDLE_EVENT_POINT(event.point, event)

    4. HANDLE_EVENT_POINT(p, event):
       a. Find segments:
          - U(p): segments with upper endpoint at p
          - L(p): segments with lower endpoint at p  
          - C(p): segments containing p in their interior

       b. If |U(p) ∪ L(p) ∪ C(p)| > 1:
          Report intersection at p

       c. Remove L(p) ∪ C(p) from status structure T

       d. Insert U(p) ∪ C(p) into status structure T

       e. Find new events:
          - If U(p) ∪ C(p) is empty:
            Check intersection of neighbors of removed segments
          - Else:
            Check leftmost segment with its left neighbor
            Check rightmost segment with its right neighbor

    5. FIND_NEW_EVENT(s1, s2, current_point):
       intersection = s1.intersect(s2)
       If intersection exists and is to the right of current_point:
           Add INTERSECTION event to Q
```

### Key Data Structures

#### 1. Event Types
- **START**: Beginning of a line segment (upper endpoint)
- **END**: Termination of a line segment (lower endpoint)  
- **INTERSECTION**: Two or more segments intersect at this point

#### 2. Status Structure
Maintains segments currently intersected by the sweep line, ordered by y-coordinate at the current x-position.

#### 3. Event Queue
Priority queue ordered by x-coordinate (then y-coordinate for ties), processing events from left to right.

## Code Structure

### Core Classes

#### `Point`
```python
@dataclass(frozen=True)
class Point:
    x: Fraction
    y: Fraction
```
- Immutable 2D point with exact arithmetic
- Lexicographic ordering for sweep line processing
- Conversion methods for visualization

#### `LineSegment`
```python
@dataclass(frozen=True)
class LineSegment:
    start: Point
    end: Point
    id: int
```
- Represents a line segment with two endpoints
- Automatically orders endpoints lexicographically
- Methods for intersection computation and point containment

#### `StatusStructure`
```python
class StatusStructure:
    def __init__(self):
        self.segments: List[LineSegment] = []
        self.current_x: Optional[Fraction] = None
```
- Maintains segments intersecting the current sweep line
- Dynamic sorting based on y-coordinates at sweep line position
- Efficient neighbor finding and segment management

#### `SweepLineIntersection`
```python
class SweepLineIntersection:
    def find_intersections(self, segments: List[LineSegment]) -> List[Tuple[Point, Set[LineSegment]]]
```
- Main algorithm implementation
- Event-driven processing with priority queue
- Handles all geometric edge cases

## Mathematical Foundation

### Line Intersection Formula

For two line segments with endpoints (x₁,y₁)-(x₂,y₂) and (x₃,y₃)-(x₄,y₄):

```
Parametric form:
Line 1: P₁ + t(P₂ - P₁) = (x₁,y₁) + t((x₂,y₂) - (x₁,y₁))
Line 2: P₃ + u(P₄ - P₃) = (x₃,y₃) + u((x₄,y₄) - (x₃,y₃))

Intersection condition:
x₁ + t(x₂-x₁) = x₃ + u(x₄-x₃)
y₁ + t(y₂-y₁) = y₃ + u(y₄-y₃)

Solution:
denominator = (x₁-x₂)(y₃-y₄) - (y₁-y₂)(x₃-x₄)

If denominator ≠ 0:
    t = ((x₁-x₃)(y₃-y₄) - (y₁-y₃)(x₃-x₄)) / denominator
    u = -((x₁-x₂)(y₁-y₃) - (y₁-y₂)(x₁-x₃)) / denominator
    
    If 0 ≤ t ≤ 1 and 0 ≤ u ≤ 1:
        Intersection exists at (x₁ + t(x₂-x₁), y₁ + t(y₂-y₁))
```

### Geometric Predicates

#### Point-on-Line Test
Uses cross product to determine if point P lies on line segment AB:
```
cross_product = (P.x - A.x)(B.y - A.y) - (P.y - A.y)(B.x - A.x)
If |cross_product| ≤ tolerance: point is on line
```

#### Segment Ordering
At sweep line position x, segments are ordered by their y-coordinates:
```python
def y_at_x(segment, x):
    if segment.is_vertical():
        return segment.start.y  # or handle specially
    t = (x - segment.start.x) / (segment.end.x - segment.start.x)
    return segment.start.y + t * (segment.end.y - segment.start.y)
```

## Algorithm Complexity Analysis

### Time Complexity: O((n + k) log n)
- **Preprocessing**: O(n²) for detecting all segment pairs, O(n log n) for initial event queue
- **Event Processing**: O((n + k) log n) where each of (n + k) events requires O(log n) operations
- **Status Structure Operations**: O(log n) per insertion/deletion/query

### Space Complexity: O(n + k)
- **Event Queue**: O(n + k) events maximum
- **Status Structure**: O(n) segments maximum at any sweep line position
- **Output**: O(k) intersection points

### Optimal Performance
This is optimal for the general case since:
- Any algorithm must examine all n segments: Ω(n)
- Any algorithm must report all k intersections: Ω(k)
- Computational geometry lower bounds suggest Ω(n log n) for segment intersection

## Edge Cases Handled

### 1. Collinear Segments
```python
# Segments sharing a common line but potentially overlapping
seg1 = LineSegment(Point(0, 0), Point(4, 0))  # Horizontal
seg2 = LineSegment(Point(2, 0), Point(6, 0))  # Overlapping horizontal
```

### 2. Vertical Segments
```python
# Special handling for infinite slope
seg = LineSegment(Point(2, 0), Point(2, 4))  # Vertical line
```

### 3. Multiple Intersections at Single Point
```python
# Three or more segments meeting at one point
segments = [
    LineSegment(Point(0, 2), Point(4, 2)),  # Horizontal
    LineSegment(Point(2, 0), Point(2, 4)),  # Vertical  
    LineSegment(Point(0, 0), Point(4, 4)),  # Diagonal
]
# All intersect at (2, 2)
```

### 4. Shared Endpoints
```python
# Segments sharing start/end points
seg1 = LineSegment(Point(0, 0), Point(2, 2))
seg2 = LineSegment(Point(2, 2), Point(4, 0))  # Shares endpoint (2,2)
```

## Usage Examples

### Basic Usage
```python
from intersection import SweepLineIntersection, LineSegment, Point

# Create line segments
segments = [
    LineSegment(Point(0, 0), Point(4, 4)),    # Diagonal
    LineSegment(Point(0, 4), Point(4, 0)),    # Diagonal
    LineSegment(Point(2, 0), Point(2, 4)),    # Vertical
]

# Find intersections
sweep_line = SweepLineIntersection()
intersections = sweep_line.find_intersections(segments)

# Display results
for point, segments_at_point in intersections:
    print(f"Intersection at ({point.x}, {point.y})")
    print(f"Segments: {[seg.id for seg in segments_at_point]}")
```

### Running Test Cases
```python
from intersection import run_test_case

# Run predefined test cases
for test_case in range(1, 4):
    run_test_case(test_case, save=True)
```

### Custom Visualization
```python
from intersection import visualize_segments_and_intersections

# Visualize results
visualize_segments_and_intersections(
    segments, intersections,
    title="Custom Intersection Analysis",
    save=True
)
```

## Test Cases

### Test Case 1: Mixed Segment Types
- **Segments**: 5 segments including horizontal and diagonal lines
- **Expected**: Multiple intersections with overlapping patterns
- **Complexity**: Tests basic algorithm functionality

### Test Case 2: Cross Pattern  
- **Segments**: 4 segments forming a symmetric cross
- **Expected**: Central intersection point where all segments meet
- **Complexity**: Tests multiple segments at single point

### Test Case 3: Triangle with Internal Segments
- **Segments**: Triangle perimeter with internal crossing segments
- **Expected**: Multiple internal and boundary intersections
- **Complexity**: Tests geometric containment and boundary cases

## Output Format

### Console Output
```
Found 2 intersection point(s):
  Intersection 1
    Exact coordinates: (2.00, 2.00)
    Segments involved: [1, 2, 3]
  Intersection 2  
    Exact coordinates: (6.67, 3.33)
    Segments involved: [2, 4]
```

### Return Format
```python
List[Tuple[Point, Set[LineSegment]]]
# Each tuple contains:
# - Point: Exact intersection coordinates (using Fraction)
# - Set[LineSegment]: All segments intersecting at this point
```

## Implementation Notes

### Exact Arithmetic Benefits
- **Precision**: Avoids floating-point errors in geometric predicates
- **Robustness**: Ensures correct handling of edge cases
- **Determinism**: Consistent results across different platforms

### Performance Optimizations
- **Event Merging**: Combines multiple events at same point
- **Lazy Evaluation**: Defers expensive computations when possible
- **Early Termination**: Skips unnecessary intersection tests

### Memory Management
- **Immutable Objects**: Points and segments are immutable for safety
- **Efficient Containers**: Uses appropriate data structures for each operation
- **Garbage Collection**: Automatic cleanup of temporary objects

## Dependencies

### Required
- `Python 3.7+`
- `fractions` (standard library)
- `dataclasses` (standard library)
- `heapq` (standard library)
- `enum` (standard library)
- `typing` (standard library)

### Optional (for visualization)
- `matplotlib`
- `numpy`

## Installation

```bash
# Clone or download the intersection.py file
# Install optional dependencies for visualization:
pip install matplotlib numpy
```

## Running the Implementation

```bash
python intersection.py
```

This will execute all test cases and generate visualization plots.

---

*Developed as part of CSC711 - Data Structures and Algorithms*  
*7th Semester Lab Assignment - Line Segment Intersection*
