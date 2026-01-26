from typing import Callable, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

class DistanceMetric(Enum):
    """Distance metric options for GIST algorithm"""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    
    def get_function(self) -> Callable:
        """Get the distance function corresponding to this metric"""
        from gist.distance import euclidean_distance, cosine_distance
        
        if self == DistanceMetric.EUCLIDEAN:
            return euclidean_distance
        elif self == DistanceMetric.COSINE:
            return cosine_distance
        else:
            raise ValueError(f"Unknown distance metric: {self}")

@dataclass
class GISTConfig:
    """Configuration for GIST algorithm"""
    k: int                    # Cardinality constraint
    lambda_param: float       # Diversity strength (λ)
    epsilon: float = 0.05     # Error parameter (ε)
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN  # Distance metric type

# Type aliases
SubmodularFunction = Callable[[Set[int], np.ndarray], float]
DistanceFunction = Callable[[np.ndarray, np.ndarray], float]
