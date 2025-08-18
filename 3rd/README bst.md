# AVL Tree Implementation

A comprehensive Python implementation of an AVL (Adelson-Velsky and Landis) Tree - a self-balancing binary search tree that ensures O(log n) time complexity for all basic operations.

## Features

- **Self-Balancing**: Automatically maintains balance through rotations
- **Efficient Operations**: O(log n) time complexity for insert, delete, and search
- **Range Queries**: Optimized range search functionality
- **Multiple Traversals**: In-order, pre-order, and post-order traversals
- **Comprehensive Statistics**: Tree metrics including height, balance factors, and rotation count
- **Validation**: Built-in AVL property validation
- **Duplicate Handling**: Configurable support for duplicate values
- **Visual Representation**: Tree structure visualization
- **Memory Efficient**: Uses `__slots__` for optimized memory usage

## Installation

No external dependencies are required for basic functionality. For enhanced tree visualization, you can optionally install:

```bash
pip install PrettyPrint
```

## Quick Start

```python
from bst import AVLTree

# Create a new AVL tree (duplicates not allowed by default)
tree = AVLTree(allow_duplicates=False)

# Insert values
tree.insert(50)
tree.insert(30)
tree.insert(70)

# Insert multiple values at once
values = [20, 40, 60, 80]
tree.insert_many(values)

# Search for a value
if tree.search(30):
    print("Value 30 found!")

# Range query
nodes_in_range = tree.range_query(25, 65)
values_in_range = [node.value for node in nodes_in_range]
print(f"Values in range [25, 65]: {values_in_range}")

# Get tree statistics
stats = tree.get_statistics()
print(f"Tree size: {stats.size}, Height: {stats.height}")

# Visualize the tree
tree.print_tree()
```

## API Reference

### Constructor

```python
AVLTree(allow_duplicates: bool)
```
- `allow_duplicates`: Whether to allow duplicate values in the tree

### Core Operations

#### Insert Operations
- `insert(value: Any) -> bool`: Insert a single value
- `insert_many(values: List[Any]) -> int`: Insert multiple values

#### Search Operations
- `search(value: Any) -> bool`: Check if value exists in tree
- `range_query(min_val: Any, max_val: Any) -> List[AVLNode]`: Find all nodes in range

#### Delete Operations
- `delete(value: Any) -> bool`: Remove a value from the tree
- `clear() -> None`: Remove all nodes from the tree

### Traversal Methods

- `inorder_traversal() -> Iterator[Any]`: Returns values in sorted order
- `preorder_traversal() -> Iterator[Any]`: Returns values in pre-order
- `postorder_traversal() -> Iterator[Any]`: Returns values in post-order

### Properties and Statistics

- `size: int`: Number of nodes in the tree
- `height: int`: Height of the tree
- `is_empty: bool`: Whether the tree is empty
- `get_statistics() -> TreeStatistics`: Comprehensive tree statistics
- `validate_avl_property() -> bool`: Validate AVL tree property

### Utility Methods

- `to_list() -> List[Any]`: Convert tree to sorted list
- `print_tree() -> None`: Visual representation of the tree

### Magic Methods

- `len(tree)`: Returns the number of nodes
- `value in tree`: Check if value exists
- `str(tree)`: String representation
- `repr(tree)`: Detailed representation

## Data Structures

### AVLNode
```python
class AVLNode:
    value: Any          # The stored value
    left: AVLNode       # Left child node
    right: AVLNode      # Right child node
    height: int         # Height of the node
```

### TreeStatistics
```python
@dataclass
class TreeStatistics:
    size: int                    # Number of nodes
    height: int                  # Tree height
    balance_factor: int          # Root balance factor
    rotations_performed: int     # Total rotations performed
```

## Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Insert | O(log n) | O(log n) |
| Delete | O(log n) | O(log n) |
| Search | O(log n) | O(log n) |
| Range Query | O(log n + k) | O(k) |
| Traversal | O(n) | O(log n) |

Where n is the number of nodes and k is the number of nodes in the result.

## Examples

### Basic Usage

```python
# Create tree and insert values
tree = AVLTree(allow_duplicates=True)
values = [49, 23, 80, 3, 10, 19, 23, 30, 37, 59, 62, 70, 80, 100, 105]
tree.insert_many(values)

print(f"Tree size: {len(tree)}")
print(f"Tree contents: {tree.to_list()}")
```

### Range Queries

```python
# Find all values between 18 and 77
nodes = tree.range_query(18, 77)
values = [node.value for node in nodes]
print(f"Values in range [18, 77]: {values}")
```

### Tree Validation

```python
# Check if tree maintains AVL property
is_valid = tree.validate_avl_property()
print(f"Is valid AVL tree: {is_valid}")

# Get detailed statistics
stats = tree.get_statistics()
print(f"Height: {stats.height}")
print(f"Rotations performed: {stats.rotations_performed}")
```

### Working with Duplicates

```python
# Allow duplicates
tree_with_dups = AVLTree(allow_duplicates=True)
tree_with_dups.insert(10)
tree_with_dups.insert(10)  # This will be inserted
print(f"Size: {len(tree_with_dups)}")  # Output: 2

# Disallow duplicates
tree_no_dups = AVLTree(allow_duplicates=False)
tree_no_dups.insert(10)
success = tree_no_dups.insert(10)  # This will be rejected
print(f"Insertion successful: {success}")  # Output: False
```

## Demo

Run the included demo to see the AVL tree in action:

```bash
python bst.py
```

The demo will:
1. Create an AVL tree with sample data
2. Display the tree structure with heights and balance factors
3. Show tree statistics
4. Perform range queries
5. Demonstrate different traversal methods
6. Show deletion operations

## Implementation Details

### AVL Property
The AVL tree maintains the invariant that for any node, the heights of its left and right subtrees differ by at most 1. This is achieved through four types of rotations:

- **Right Rotation**: Used for Left-Left case
- **Left Rotation**: Used for Right-Right case
- **Left-Right Rotation**: Used for Left-Right case
- **Right-Left Rotation**: Used for Right-Left case

### Range Query Optimization
The range query implementation uses an optimized traversal that only visits nodes that could potentially be in the range, achieving better than O(n) performance in most cases.

### Memory Optimization
The `AVLNode` class uses `__slots__` to reduce memory overhead by preventing the creation of `__dict__` for each instance.

## Author

Created as part of CSC711 Data Structures and Algorithms coursework.
