def idw_interpolation(points, values, grid_x, grid_y, power=2):
    """
    Perform Inverse Distance Weighting (IDW) interpolation.

    Parameters:
    points : array-like
        Coordinates of the known data points (e.g., rain gauges).
    values : array-like
        Values at the known data points (e.g., rainfall measurements).
    grid_x : array-like
        X-coordinates of the grid where interpolation is to be performed.
    grid_y : array-like
        Y-coordinates of the grid where interpolation is to be performed.
    power : float
        The power parameter for IDW, controlling the influence of nearby points.

    Returns:
    grid_values : array
        Interpolated values on the grid.
    """
    from scipy.spatial import cKDTree
    import numpy as np

    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(points)
    
    # Prepare an array to hold the interpolated values
    grid_values = np.zeros((len(grid_y), len(grid_x)))

    for i, (x, y) in enumerate(zip(grid_x, grid_y)):
        # Find the k nearest neighbors
        distances, indices = tree.query((x, y), k=len(points))
        
        # Avoid division by zero
        distances = np.where(distances == 0, 1e-10, distances)
        
        # Calculate weights
        weights = 1 / distances**power
        
        # Normalize weights
        weights /= weights.sum()
        
        # Interpolate the value
        grid_values[i // len(grid_x), i % len(grid_x)] = np.dot(weights, values[indices])

    return grid_values


def create_grid(x_min, x_max, y_min, y_max, resolution):
    """
    Create a grid of points for interpolation.

    Parameters:
    x_min : float
        Minimum x-coordinate.
    x_max : float
        Maximum x-coordinate.
    y_min : float
        Minimum y-coordinate.
    y_max : float
        Maximum y-coordinate.
    resolution : float
        Distance between grid points.

    Returns:
    grid_x : array
        X-coordinates of the grid points.
    grid_y : array
        Y-coordinates of the grid points.
    """
    import numpy as np

    grid_x = np.arange(x_min, x_max, resolution)
    grid_y = np.arange(y_min, y_max, resolution)

    return np.meshgrid(grid_x, grid_y)