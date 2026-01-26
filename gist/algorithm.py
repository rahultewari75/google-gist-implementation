import numpy as np
import pandas as pd
from typing import Set, List, Callable, Tuple, Optional
from gist.types import GISTConfig
from gist.distance import get_distance_function, compute_pairwise_distances
from gist.submodular import div

class GIST:
    """
    GIST: Greedy Independent Set Thresholding for Max-Min Diversification
    with Monotone Submodular Utility
    
    Implements Algorithm 1 from the paper with 1/2-approximation guarantee.
    """
    
    def __init__(self, 
                 config: GISTConfig,
                 submodular_fn: Callable,
                 distance_fn: Optional[Callable] = None):
        self.config = config
        self.submodular_fn = submodular_fn
        self.distance_fn = distance_fn or config.distance_metric.get_function()
        self.data = None
        self.dist_matrix = None
    
    def greedy_independent_set(self, 
                               data: np.ndarray,
                               dist_matrix: np.ndarray,
                               d: float) -> Set[int]:
        """
        GreedyIndependentSet subroutine (Algorithm 1, lines 12-19)
        
        Builds a maximal independent set of G_d(V) by greedily adding points
        with highest marginal gain subject to distance constraint.
        
        Args:
            data: Array of points (n x d)
            dist_matrix: Precomputed pairwise distance matrix
            d: Distance threshold
            
        Returns:
            Set of selected indices
        """
        S = set()
        
        for _ in range(self.config.k):
            # Find candidates: points v where dist(v, S) >= d
            candidates = []
            for v in range(len(data)):
                if v in S:
                    continue
                
                # Check distance constraint
                if not S:
                    # First point, always valid
                    candidates.append(v)
                else:
                    min_dist_to_S = min(dist_matrix[v, s] for s in S)
                    if min_dist_to_S >= d:
                        candidates.append(v)
            
            if not candidates:
                # S is a maximal independent set
                break
            
            # Select candidate with highest marginal gain
            best_v = None
            best_gain = float('-inf')
            
            for v in candidates:
                gain = self._marginal_gain(v, S, data)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v is not None:
                S.add(best_v)
        
        return S
    
    def _marginal_gain(self, v: int, S: Set[int], data: np.ndarray) -> float:
        """Compute marginal gain g(v | S)"""
        if hasattr(self.submodular_fn, 'marginal_gain'):
            return self.submodular_fn.marginal_gain(v, S, data)
        else:
            # Fallback: compute difference
            return (self.submodular_fn(S | {v}, data) - 
                   self.submodular_fn(S, data))
    
    def _objective(self, S: Set[int], data: np.ndarray, 
                  dist_matrix: np.ndarray) -> float:
        """
        Objective function: f(S) = g(S) + λ·div(S)
        """
        g_value = self.submodular_fn(S, data)
        div_value = div(S, dist_matrix)
        return g_value + self.config.lambda_param * div_value
    
    def fit(self, df: pd.DataFrame) -> 'GIST':
        """
        Fit GIST on a pandas DataFrame.
        
        Args:
            df: DataFrame where rows are points and columns are features
            
        Returns:
            self for method chaining
        """
        self.data = df.values
        self.dist_matrix = compute_pairwise_distances(
            self.data, self.distance_fn
        )
        return self
    
    def select(self) -> Set[int]:
        """
        Run GIST algorithm and return selected subset.
        
        Implements Algorithm 1 from the paper.
        
        Returns:
            Set of selected indices
        """
        if self.data is None:
            raise ValueError("Must call fit() before select()")
        
        data = self.data
        dist_matrix = self.dist_matrix
        k = self.config.k
        epsilon = self.config.epsilon
        
        # Line 2: Initialize with greedy solution (d=0)
        S = self.greedy_independent_set(data, dist_matrix, d=0.0)
        best_value = self._objective(S, data, dist_matrix)
        
        # Lines 3-6: Consider two diametrical points
        d_max = np.max(dist_matrix)
        # Find two points with maximum distance
        max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        diametrical_pair = {max_idx[0], max_idx[1]}
        if len(diametrical_pair) == 2 and k >= 2:
            pair_value = self._objective(diametrical_pair, data, dist_matrix)
            if pair_value > best_value:
                S = diametrical_pair
                best_value = pair_value
        
        # Lines 7-11: Iterate over distance thresholds
        # D = {(1+ε)^i · ε·d_max/2 : (1+ε)^i ≤ 2/ε, i ∈ Z≥0}
        thresholds = []
        i = 0
        while True:
            threshold = ((1 + epsilon) ** i) * (epsilon * d_max / 2)
            if (1 + epsilon) ** i > 2 / epsilon:
                break
            thresholds.append(threshold)
            i += 1
        
        # For each threshold, run GreedyIndependentSet
        for d in thresholds:
            T = self.greedy_independent_set(data, dist_matrix, d)
            T_value = self._objective(T, data, dist_matrix)
            
            if T_value >= best_value:
                S = T
                best_value = T_value
        
        return S
    
    def get_selected_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with selected rows"""
        selected_indices = self.select()
        return df.iloc[list(selected_indices)]
