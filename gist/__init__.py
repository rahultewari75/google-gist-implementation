"""
GIST: Greedy Independent Set Thresholding for Max-Min Diversification
with Monotone Submodular Utility

Implementation of the GIST algorithm from:
"GIST: Greedy Independent Set Thresholding for Max-Min Diversification 
with Submodular Utility" (Google, 2024)
"""

from gist.algorithm import GIST
from gist.types import GISTConfig, DistanceMetric
from gist.submodular import LinearUtility, CoverageUtility
from gist.distance import get_distance_function

__all__ = [
    'GIST',
    'GISTConfig',
    'DistanceMetric',
    'LinearUtility',
    'CoverageUtility',
    'get_distance_function',
]
