import numpy as np
import pandas as pd
from typing import Set, List, Callable, Tuple
from itertools import combinations
from gist.algorithm import GIST
from gist.types import GISTConfig

def compute_div(S: Set[int], dist_matrix: np.ndarray) -> float:
    """Compute max-min diversity"""
    if len(S) <= 1:
        return np.max(dist_matrix)
    
    S_list = list(S)
    min_dist = float('inf')
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            min_dist = min(min_dist, dist_matrix[S_list[i], S_list[j]])
    return min_dist

def brute_force_optimal(data: np.ndarray,
                       dist_matrix: np.ndarray,
                       submodular_fn: Callable,
                       config: GISTConfig) -> Tuple[Set[int], float]:
    """
    Brute force solver to find optimal solution.
    Enumerates all subsets of size <= k.
    
    Returns:
        (optimal_set, optimal_value)
    """
    n = len(data)
    k = config.k
    
    best_set = None
    best_value = float('-inf')
    
    # Enumerate all subsets of size 1 to k
    for size in range(1, min(k + 1, n + 1)):
        for subset in combinations(range(n), size):
            S = set(subset)
            
            # Compute objective
            g_value = submodular_fn(S, data)
            div_value = compute_div(S, dist_matrix)
            value = g_value + config.lambda_param * div_value
            
            if value > best_value:
                best_value = value
                best_set = S
    
    return best_set, best_value

def generate_test_dataset(n: int, d: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate random test dataset.
    
    Args:
        n: Number of points
        d: Number of dimensions
        seed: Random seed
        
    Returns:
        DataFrame with n rows and d columns
    """
    np.random.seed(seed)
    data = np.random.randn(n, d)
    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(d)])

def generate_submodular_breaking_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset with 4+ dimensions to potentially break submodularity
    of the combined objective (though g(S) itself remains submodular).
    """
    return generate_test_dataset(n, d=4, seed=seed)
