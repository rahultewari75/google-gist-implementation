import numpy as np
from typing import Callable, Union
from gist.types import DistanceMetric

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two points"""
    return np.linalg.norm(x - y)

def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance: 1 - cosine_similarity"""
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 1.0
    cosine_sim = dot_product / (norm_x * norm_y)
    return 1.0 - cosine_sim

def get_distance_function(metric: Union[DistanceMetric, str]) -> Callable:
    """
    Factory function to get distance metric.
    
    Args:
        metric: DistanceMetric enum or string (for backward compatibility)
        
    Returns:
        Distance function
    """
    # Handle enum
    if isinstance(metric, DistanceMetric):
        return metric.get_function()
    
    # Handle string (backward compatibility)
    if isinstance(metric, str):
        metrics = {
            'euclidean': euclidean_distance,
            'cosine': cosine_distance,
        }
        return metrics.get(metric, euclidean_distance)
    
    raise TypeError(f"metric must be DistanceMetric enum or str, got {type(metric)}")

def compute_pairwise_distances(data: np.ndarray, 
                               dist_fn: Callable) -> np.ndarray:
    """Precompute pairwise distance matrix"""
    n = len(data)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = dist_fn(data[i], data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix
