from typing import Optional, List, Iterator, Tuple, Any
from dataclasses import dataclass

@dataclass
class TreeStatistics:
    """Statistics about the tree structure and operations."""
    size: int
    height: int
    balance_factor: int
    rotations_performed: int

class AVLNode:
    """
    Node class for AVL Tree.
    Each node contains a value, references to left and right children,
    and height information for balancing operations.
    
    """
    __slots__ = ['value', 'left', 'right', 'height']

    def __init__(self, value: Any) -> None:
        """
        Initialize a new AVL node.
        
        Args:
            value: The value to store in this node
        """
        self.value = value
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height: int = 1
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return f"AVLNode({self.value})"
    
class AVLTree:
    """
    Self-balancing Binary Search Tree (AVL Tree) implementation.
    
    An AVL tree is a self-balancing binary search tree where the heights
    of the two child subtrees of any node differ by at most one. This
    ensures O(log n) time complexity for all basic operations.
    
    Features:
    - Automatic balancing through rotations
    - Range queries with optimal time complexity
    - In-order, pre-order, and post-order traversals
    - Comprehensive statistics and validation
    - Memory-efficient implementation
    
    Example:
        >>> tree = AVLTree()
        >>> tree.insert_many([50, 30, 70, 20, 40, 60, 80])
        >>> nodes_in_range = tree.range_query(18, 77)
        >>> print([node.value for node in nodes_in_range])
        [20, 30, 40, 50, 60, 70]
    """

    def __init__(self, allow_duplicates: bool) -> None:
        """Initialize an empty AVL tree."""
        self.root: Optional[AVLNode] = None
        self._size: int = 0
        self._rotation_count: int = 0
        self.allow_duplicates = allow_duplicates

    @property
    def size(self) -> int:
        """Get the number of nodes in the tree."""
        return self._size
    
    @property
    def height(self) -> int:
        """Get the height of the tree."""
        return self._get_height(self.root)
    
    @property
    def is_empty(self) -> bool:
        """Check if the tree is empty."""
        return self.root is None

    def _get_height(self, node: Optional[AVLNode]) -> int:
        """
        Get the height of a given node.
        
        Args:
            node: The node to get height for
            
        Returns:
            Height of the node (0 for None)
        """
        return 0 if node is None else node.height
    
    def _update_height(self, node: AVLNode) -> None:
        """
        Update the height of a node based on its children.
        
        Args:
            node: The node to update
        """
        node.height = 1 + max(
            self._get_height(node.left),
            self._get_height(node.right)
        )
    
    def _get_balance_factor(self, node: Optional[AVLNode]) -> int:
        """
        Calculate the balance factor of a node.
        
        Args:
            node: The node to calculate balance factor for
            
        Returns:
            Balance factor (left_height - right_height)
        """
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_right(self, y: AVLNode) -> AVLNode:
        """
        Perform a right rotation.
        
        Args:
            y: The node to rotate around
            
        Returns:
            New root of the subtree after rotation
        """
        x = y.left
        t2 = x.right
        
        # Perform rotation
        x.right = y # type: ignore
        y.left = t2
        
        # Update heights
        self._update_height(y)
        self._update_height(x)
        
        self._rotation_count += 1
        return x # type: ignore
    
    def _rotate_left(self, x: AVLNode) -> AVLNode:
        """
        Perform a left rotation

        Args:
            x: The node to rotate around
        Returns:
            New root of the subtree after rotation
        """
        y = x.right
        t2 = y.left

        # Perform rotation
        y.left = x # type: ignore
        x.right = t2

        # Update heights
        self._update_height(x)
        self._update_height(y)

        self._rotation_count += 1
        return y
    
    def insert(self, value: Any) -> bool:
        """
        Insert a value into the tree.
        
        Args:
            value: The value to insert
            
        Returns:
            True if the value was inserted, False if it already exists
        """
        initial_size = self._size
        self.root = self._insert_recursive(self.root, value)
        return self._size > initial_size
    
    def _insert_recursive(self, node: Optional[AVLNode], value: Any) -> AVLNode:
        """
        Recursively insert a value and maintain AVL property.
        
        Args:
            node: Current node in recursion
            value: Value to insert
            
        Returns:
            Root of the updated subtree
        """
        # Standard BST insertion
        if node is None:
            self._size += 1
            return AVLNode(value)
        
        if not self.allow_duplicates:
            if value < node.value:
                node.left = self._insert_recursive(node.left, value)
            elif value > node.value:
                node.right = self._insert_recursive(node.right, value)
            else:
                # Duplicate values are not allowed
                return node
        else:
            if value <= node.value:
                node.left = self._insert_recursive(node.left, value)
            else:
                node.right = self._insert_recursive(node.right, value)

        # Update height
        self._update_height(node)
        
        # Get balance factor and perform rotations if needed
        balance = self._get_balance_factor(node)
        
        # Left Left Case
        if balance > 1 and value < node.left.value:
            return self._rotate_right(node)
        
        # Right Right Case
        if balance < -1 and value > node.right.value:
            return self._rotate_left(node)
        
        # Left Right Case
        if balance > 1 and value > node.left.value:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right Left Case
        if balance < -1 and value < node.right.value:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def delete(self, value: Any) -> bool:
        """
        Delete a value from the tree.
        
        Args:
            value: The value to delete
            
        Returns:
            True if the value was deleted, False if it wasn't found
        """
        initial_size = self._size
        self.root = self._delete_recursive(self.root, value)
        return self._size < initial_size
    
    def _delete_recursive(self, node: Optional[AVLNode], value: Any) -> Optional[AVLNode]:
        """
        Recursively delete a value and maintain AVL property.
        
        Args:
            node: Current node in recursion
            value: Value to delete
            
        Returns:
            Root of the updated subtree
        """
        if node is None:
            return None
        
        # Standard BST deletion
        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            self._size -= 1
            
            # Node to be deleted found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            
            # Node with two children
            successor = self._find_min_node(node.right)
            node.value = successor.value
            node.right = self._delete_recursive(node.right, successor.value)
            self._size += 1  # Adjust since we'll decrement again in recursion
        
        # Update height
        self._update_height(node)
        
        # Get balance factor and perform rotations if needed
        balance = self._get_balance_factor(node)
        
        # Left Left Case
        if balance > 1 and self._get_balance_factor(node.left) >= 0:
            return self._rotate_right(node)
        
        # Left Right Case
        if balance > 1 and self._get_balance_factor(node.left) < 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right Right Case
        if balance < -1 and self._get_balance_factor(node.right) <= 0:
            return self._rotate_left(node)
        
        # Right Left Case
        if balance < -1 and self._get_balance_factor(node.right) > 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def _find_min_node(self, node: AVLNode) -> AVLNode:
        """Find the node with minimum value in a subtree."""
        while node.left is not None:
            node = node.left
        return node

    def _find_max_node(self, node: AVLNode) -> AVLNode:
        """Find the node with maximum value in a subtree."""
        while node.right is not None:
            node = node.right
        return node
    
    def search(self, value: Any) -> bool:
        """
        Search for a value in the tree.
        
        Args:
            value: The value to search for
            
        Returns:
            True if the value exists, False otherwise
        """
        return self._search_recursive(self.root, value) is not None
    
    def _search_recursive(self, node: Optional[AVLNode], value: Any) -> Optional[AVLNode]:
        """Recursively search for a value."""
        if node is None or node.value == value:
            return node
        
        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)
    
    def range_query(self, min_val: Any, max_val: Any) -> List[AVLNode]:
        """
        Find all nodes with values in the given range [min_val, max_val].
        
        This method performs an optimized range traversal that only visits
        nodes that could potentially be in the range, achieving better than
        O(n) performance in most cases.
        
        Args:
            min_val: Minimum value of the range (inclusive)
            max_val: Maximum value of the range (inclusive)
            
        Returns:
            List of nodes with values in the specified range, sorted in ascending order
            
        Example:
            >>> tree = AVLTree()
            >>> tree.insert_many([10, 5, 15, 2, 7, 12, 20])
            >>> nodes = tree.range_query(5, 15)
            >>> [node.value for node in nodes]
            [5, 7, 10, 12, 15]
        """
        if min_val > max_val:
            raise ValueError("min_val cannot be greater than max_val")

        result: List[AVLNode] = []
        self._range_query_recursive(self.root, min_val, max_val, result)
        return result
    
    def _range_query_recursive(self, node: Optional[AVLNode], min_val: Any, 
                             max_val: Any, result: List[AVLNode]) -> None:
        """
        Recursively find nodes in range using optimized traversal.
        
        Args:
            node: Current node
            min_val: Minimum value of range
            max_val: Maximum value of range
            result: List to store results
        """
        if node is None:
            return
        
        # If current node is greater than min_val, explore left subtree
        if node.value > min_val:
            self._range_query_recursive(node.left, min_val, max_val, result)
        
        # If current node is in range, add it to result
        if min_val <= node.value <= max_val:
            result.append(node)
        
        # If current node is less than max_val, explore right subtree
        if node.value < max_val:
            self._range_query_recursive(node.right, min_val, max_val, result)

    def insert_many(self, values: List[Any]) -> int:
        """
        Insert multiple values into the tree.
        
        Args:
            values: List of values to insert
            
        Returns:
            Number of values successfully inserted
        """
        count = 0
        for value in values:
            if self.insert(value):
                count += 1
        return count
    
    def inorder_traversal(self) -> Iterator[Any]:
        """
        Perform in-order traversal of the tree.
        
        Yields:
            Values in sorted order
        """
        yield from self._inorder_recursive(self.root)   

    def _inorder_recursive(self, node: Optional[AVLNode]) -> Iterator[Any]:
        """Recursively perform in-order traversal."""
        if node is not None:
            yield from self._inorder_recursive(node.left)
            yield node.value
            yield from self._inorder_recursive(node.right)
    
    def preorder_traversal(self) -> Iterator[Any]:
        """
        Perform pre-order traversal of the tree.
        
        Yields:
            Values in pre-order
        """
        yield from self._preorder_recursive(self.root)
    
    def _preorder_recursive(self, node: Optional[AVLNode]) -> Iterator[Any]:
        """Recursively perform pre-order traversal."""
        if node is not None:
            yield node.value
            yield from self._preorder_recursive(node.left)
            yield from self._preorder_recursive(node.right)
    
    def postorder_traversal(self) -> Iterator[Any]:
        """
        Perform post-order traversal of the tree.
        
        Yields:
            Values in post-order
        """
        yield from self._postorder_recursive(self.root)
    
    def _postorder_recursive(self, node: Optional[AVLNode]) -> Iterator[Any]:
        """Recursively perform post-order traversal."""
        if node is not None:
            yield from self._postorder_recursive(node.left)
            yield from self._postorder_recursive(node.right)
            yield node.value
    
    def get_statistics(self) -> TreeStatistics:
        """
        Get comprehensive statistics about the tree.
        
        Returns:
            TreeStatistics object with size, height, balance factor, and rotation count
        """
        return TreeStatistics(
            size=self._size,
            height=self.height,
            balance_factor=self._get_balance_factor(self.root),
            rotations_performed=self._rotation_count
        )

    def validate_avl_property(self) -> bool:
        """
        Validate that the tree maintains the AVL property.
        
        Returns:
            True if the tree is a valid AVL tree, False otherwise
        """
        return self._validate_avl_recursive(self.root)[0]
    
    def _validate_avl_recursive(self, node: Optional[AVLNode]) -> Tuple[bool, int]:
        """
        Recursively validate AVL property and return (is_valid, height).
        
        Returns:
            Tuple of (is_valid_avl, actual_height)
        """
        if node is None:
            return True, 0
        
        # Validate left and right subtrees
        left_valid, left_height = self._validate_avl_recursive(node.left)
        right_valid, right_height = self._validate_avl_recursive(node.right)
        
        # Check if current node maintains AVL property
        balance_factor = left_height - right_height
        is_balanced = abs(balance_factor) <= 1
        
        # Check if stored height is correct
        expected_height = 1 + max(left_height, right_height)
        height_correct = node.height == expected_height
        
        is_valid = left_valid and right_valid and is_balanced and height_correct
        
        return is_valid, expected_height
    
    def clear(self) -> None:
        """Clear all nodes from the tree."""
        self.root = None
        self._size = 0
        self._rotation_count = 0

    def to_list(self) -> List[Any]:
        """
        Convert tree to a sorted list.
        
        Returns:
            List of all values in sorted order
        """
        return list(self.inorder_traversal())
    
    def print_tree_fallback(self, node: Optional[AVLNode] = None, 
                   level: int = 0, prefix: str = "Root: ") -> None:
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
            print(" " * (level * 4) + prefix + str(node.value) + 
                  f" (h:{node.height}, bf:{self._get_balance_factor(node)})")
            
            if node.left is not None or node.right is not None:
                if node.left:
                    self.print_tree_fallback(node.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- None")
                
                if node.right:
                    self.print_tree_fallback(node.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- None")
    
    def print_tree(self) -> None:
        """
        Print a visual representation of the AVL tree with height and balance factor using PrettyPrintTree.
        """
        try:
            from PrettyPrint import PrettyPrintTree

            if self.root is None:
                print("Tree is empty")
                return

            get_children = lambda node: [child for child in [node.left, node.right] if child is not None]
            bf = lambda node: self._get_balance_factor(node)
            get_value = lambda node: f"{node.value}({node.height}, {bf(node)})"

            pt = PrettyPrintTree(get_children, get_value)
            pt(self.root)

        except ImportError:
            self.print_tree_fallback(self.root)

    def __len__(self) -> int:
        """Return the number of nodes in the tree."""
        return self._size
    
    def __contains__(self, value: Any) -> bool:
        """Check if a value exists in the tree."""
        return self.search(value)
    
    def __str__(self) -> str:
        """String representation of the tree."""
        if self.is_empty:
            return "AVLTree(empty)"
        return f"AVLTree({list(self.inorder_traversal())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"AVLTree(size={self._size}, height={self.height}, "
                f"rotations={self._rotation_count})")
    

def demo_range_query() -> None:
    """
    Demonstrate the AVL tree with range query functionality.
    
    This function creates a sample tree and performs range queries
    as specified in the requirements.
    """
    print("AVL Tree Range Query Demonstration")
    print("=" * 50)
    
    # Create tree and insert sample values
    tree = AVLTree(allow_duplicates=True)
    sample_values = [49, 23, 80, 3, 10, 19, 23, 30, 37, 59, 62, 70, 80, 100, 105, 3, 19, 30, 49, 59, 70, 89, 100, 10, 37, 62, 89 ]
    sample_values.sort(reverse=True) # Not required, but to replicate given question's order
    print(f"Inserting values: {sample_values}")
    inserted_count = tree.insert_many(sample_values)
    print(f"Successfully inserted {inserted_count} values")
    print()
    
    # Display tree structure
    print("Tree Structure:")
    print("-" * 15)
    tree.print_tree()
    print()
    
    # Display tree statistics
    stats = tree.get_statistics()
    print(f"Tree Statistics:")
    print(f"  Size: {stats.size}")
    print(f"  Height: {stats.height}")
    print(f"  Root Balance Factor: {stats.balance_factor}")
    print(f"  Rotations Performed: {stats.rotations_performed}")
    print(f"  AVL Property Valid: {tree.validate_avl_property()}")
    print()
    

    print("Range Query [18:77]:")
    print("-" * 20)
    
    range_nodes = tree.range_query(18, 77)
    range_values = [node.value for node in range_nodes]
    
    print(f"Nodes in range [18, 77]: {range_values}")
    print(f"Number of nodes found: {len(range_values)}")
    print()
    
    # Additional range queries for demonstration
    test_ranges = [(25, 65), (40, 60), (0, 100), (90, 100)]
    
    print("Additional Range Queries:")
    print("-" * 25)
    
    for min_val, max_val in test_ranges:
        nodes = tree.range_query(min_val, max_val)
        values = [node.value for node in nodes]
        print(f"Range [{min_val}, {max_val}]: {values}")
    
    print()
    
    # Show different traversals
    print("Tree Traversals:")
    print("-" * 15)
    print(f"In-order:    {list(tree.inorder_traversal())}")
    print(f"Pre-order:   {list(tree.preorder_traversal())}")
    print(f"Post-order:  {list(tree.postorder_traversal())}")

    # Deleting nodes

    del_int = 49
    print(f"\nDeleting value: {del_int}")
    if tree.delete(del_int):
        print(f"Successfully deleted {del_int}")
        # Tree statistics after deletion
        stats = tree.get_statistics()
        print(f"Tree Statistics after deletion:")
        print(f"  Size: {stats.size}")
        print(f"  Height: {stats.height}")
        print(f"  Root Balance Factor: {stats.balance_factor}")
        tree.print_tree()

if __name__ == "__main__":
    demo_range_query()

    
