import pytest
import numpy as np
import pandas as pd
from gist.algorithm import GIST
from gist.types import GISTConfig, DistanceMetric
from gist.submodular import LinearUtility

def test_gist_initialization():
    """Test GIST initialization"""
    config = GISTConfig(k=5, lambda_param=1.0)
    np.random.seed(42)
    submodular_fn = LinearUtility(np.random.rand(10))
    gist = GIST(config, submodular_fn)
    assert gist.config == config
    assert gist.submodular_fn == submodular_fn

def test_greedy_independent_set():
    """Test GreedyIndependentSet subroutine"""
    n, k = 15, 5
    np.random.seed(42)
    data = np.random.randn(n, 3)
    df = pd.DataFrame(data)
    
    config = GISTConfig(k=k, lambda_param=1.0, distance_metric=DistanceMetric.EUCLIDEAN)
    weights = np.random.rand(n)
    submodular_fn = LinearUtility(weights)
    
    gist = GIST(config, submodular_fn)
    gist.fit(df)
    
    # Test with d=0 (should select k points greedily)
    solution = gist.greedy_independent_set(data, gist.dist_matrix, d=0.0)
    assert len(solution) <= k
    assert len(solution) > 0

def test_objective_function():
    """Test objective function computation"""
    n = 10
    np.random.seed(42)
    data = np.random.randn(n, 3)
    df = pd.DataFrame(data)
    
    config = GISTConfig(k=5, lambda_param=1.0, distance_metric=DistanceMetric.EUCLIDEAN)
    weights = np.random.rand(n)
    submodular_fn = LinearUtility(weights)
    
    gist = GIST(config, submodular_fn)
    gist.fit(df)
    
    S = {0, 1, 2}
    value = gist._objective(S, data, gist.dist_matrix)
    assert value > 0  # Should be positive
