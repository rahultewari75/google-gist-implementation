import numpy as np
from typing import Set, Callable

class LinearUtility:
    """Linear utility function: g(S) = sum_{v in S} w(v)"""
    def __init__(self, weights: np.ndarray):
        self.weights = weights
    
    def __call__(self, S: Set[int], data: np.ndarray) -> float:
        return sum(self.weights[i] for i in S)
    
    def marginal_gain(self, v: int, S: Set[int], data: np.ndarray) -> float:
        """Marginal gain of adding v to S"""
        return self.weights[v]

class CoverageUtility:
    """Coverage utility: g(S) = sum_{i} max_{j in S} c_{ij}"""
    def __init__(self, coverage_matrix: np.ndarray):
        # coverage_matrix[i, j] = value of covering item i with point j
        self.C = coverage_matrix
    
    def __call__(self, S: Set[int], data: np.ndarray) -> float:
        if not S:
            return 0.0
        return np.sum(np.max(self.C[:, list(S)], axis=1))
    
    def marginal_gain(self, v: int, S: Set[int], data: np.ndarray) -> float:
        """Marginal gain of adding v to S"""
        if not S:
            return np.sum(self.C[:, v])
        current_max = np.max(self.C[:, list(S)], axis=1) if S else np.zeros(self.C.shape[0])
        new_max = np.maximum(current_max, self.C[:, v])
        return np.sum(new_max - current_max)

def div(S: Set[int], dist_matrix: np.ndarray) -> float:
    """
    Max-min diversity: div(S) = min_{u,v in S, u≠v} dist(u,v)
    Returns diameter if |S| <= 1
    """
    if len(S) <= 1:
        # Return maximum distance in entire dataset (diameter)
        return np.max(dist_matrix)
    
    S_list = list(S)
    min_dist = float('inf')
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            dist = dist_matrix[S_list[i], S_list[j]]
            min_dist = min(min_dist, dist)
    return min_dist
