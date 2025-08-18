# KD-Tree Range Search – Short Code Report

Date: 2025-08-11
File: `884_Ass4.py`

## Summary
- Implements a 2D KD-tree with:
  - Recursive build over alternating axes (x, y)
  - Range search SEARCHKDTREE(ν, R)
  - Text/tree printing (PrettyPrint optional) and Matplotlib visualization
- Includes a `test_kdtree()` harness with sample points and multiple range queries.

## Key Data Structures / API
- `Point(x, y)` – simple 2D point.
- `KDNode(point, left, right, axis)` – axis ∈ {0:x, 1:y}.
- `KDTree`:
  - `build(points)` → builds tree from list of [x, y] or Point
  - `range_search(R, start_node=None)` → returns points in axis-aligned rectangle R = [[x_min, y_min], [x_max, y_max]]
  - `print_tree()` / fallback printer; `visualize(query_ranges=None)`

## Pseudocode (core)

Build KD-tree (recursive):
```
BUILD(points, depth):
  if points empty: return null
  axis ← depth mod 2   # 0=x, 1=y
  sort points by axis
  m ← ⌊len(points)/2⌋
  node.point ← points[m]; node.axis ← axis
  node.left  ← BUILD(points[0:m], depth+1)
  node.right ← BUILD(points[m+1: ], depth+1)
  return node
```

Range search SEARCHKDTREE(ν, R):
```
SEARCH(node, R, depth):
  if node = null: return []
  axis ← depth mod 2; p ← node.point
  results ← []
  if p inside R: append p to results

  if axis = 0:  # split on x at p.x
    if R.x_min ≤ p.x: results += SEARCH(node.left,  R, depth+1)
    if R.x_max ≥ p.x: results += SEARCH(node.right, R, depth+1)
  else:         # split on y at p.y
    if R.y_min ≤ p.y: results += SEARCH(node.left,  R, depth+1)
    if R.y_max ≥ p.y: results += SEARCH(node.right, R, depth+1)

  return results
```

## Complexity
- Build (as implemented: sorting at each recursion):
  - T(n) = n log n + 2 T(n/2) ⇒ O(n log^2 n) average
  - With linear-time median selection per level: O(n log n)
- Range search:
  - Expected O(log n + k) in balanced trees for typical ranges (k = results)
  - Worst case O(n)
- Space:
  - O(n) nodes; recursion depth O(log n) average, O(n) worst

## Design/Behavior Notes
- Axis alternates by depth; median chosen by sorted order, left/right exclude the median element.
- Duplicates on the splitting coordinate go left or right based on sort stability and position; one median point is kept at the node.
- Search explores a child iff R intersects that half-space relative to the splitting line.
- Logging: build steps stored in `build_log` and printed; search prints traversal decisions.
- Visualization: draws points, alternating partition lines, an approximate tree diagram, and optional query rectangles (Matplotlib required).

## Sample I/O (from `test_kdtree()`)
Input points:
- [[1,9],[2,3],[4,1],[3,7],[5,4],[6,8],[7,2],[8,8],[7,9],[9,6]]

Example ranges tested:
- R1 = [[3,7],[7,9]]  → upper region; prints visited nodes and returns matching points
- R2 = [[1,1],[6,8]]  → lower-left quadrant
- R3 = [[6,6],[9,9]]  → upper-right region
- R4 = [[0,0],[10,10]] → entire space (returns all points)

Outputs include:
- Console logs of build partitions and search path
- Optional tree/partitions plot if Matplotlib is available

## Limitations / Improvements
- Build can be optimized to O(n log n) by using nth_element/quickselect to find medians without full sort at each level.
- Range search could use stored bounding rectangles per node to perform full-subtree inclusion pruning (report whole subtree) for speed.
- Tie-breaking: consider stable, explicit rule for equal keys to improve balance with many duplicates.
- Visualization partition extents depend on current axis limits; could compute exact segment extents from subtree bounds for clarity.

---
Concise takeaway: Correct and readable KD-tree with clear logging and visualization; suitable for demonstrations, with straightforward paths for performance and robustness improvements.
