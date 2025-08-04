
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
