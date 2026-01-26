import pytest
import numpy as np
import pandas as pd
from gist.algorithm import GIST
from gist.types import GISTConfig, DistanceMetric
from gist.submodular import LinearUtility, CoverageUtility
from gist.distance import compute_pairwise_distances
from tests.test_utils import (
    brute_force_optimal, 
    generate_test_dataset,
    generate_submodular_breaking_dataset
)

class TestApproximationGuarantee:
    """Test suite to validate 1/2-approximation guarantee"""
    
    def test_linear_utility_small(self):
        """Test with linear utility on small dataset"""
        n, k = 20, 5
        df = generate_test_dataset(n, d=10, seed=42)
        data = df.values
        
        # Create linear utility with random weights
        np.random.seed(42)
        weights = np.random.rand(n)
        submodular_fn = LinearUtility(weights)
        
        config = GISTConfig(k=k, lambda_param=1.0, epsilon=0.05, distance_metric=DistanceMetric.EUCLIDEAN)
        dist_fn = DistanceMetric.EUCLIDEAN.get_function()
        dist_matrix = compute_pairwise_distances(data, dist_fn)
        
        # Run GIST
        gist = GIST(config, submodular_fn)
        gist.fit(df)
        gist_solution = gist.select()
        gist_value = gist._objective(gist_solution, data, dist_matrix)
        
        # Brute force optimal
        optimal_set, optimal_value = brute_force_optimal(
            data, dist_matrix, submodular_fn, config
        )
        
        # Check approximation ratio
        approximation_ratio = gist_value / optimal_value
        assert approximation_ratio >= 0.5, \
            f"GIST achieved {approximation_ratio:.4f}, expected >= 0.5"
        
        print(f"Linear utility test: ratio = {approximation_ratio:.4f}")
    
    def test_coverage_utility_medium(self):
        """Test with coverage utility on medium dataset"""
        n, k = 50, 10
        df = generate_test_dataset(n, d=15, seed=123)
        data = df.values
        
        # Create coverage utility
        m = 30  # number of items to cover
        np.random.seed(123)
        coverage_matrix = np.random.rand(m, n)
        submodular_fn = CoverageUtility(coverage_matrix)
        
        config = GISTConfig(k=k, lambda_param=0.5, epsilon=0.05, distance_metric=DistanceMetric.EUCLIDEAN)
        dist_fn = DistanceMetric.EUCLIDEAN.get_function()
        dist_matrix = compute_pairwise_distances(data, dist_fn)
        
        # Run GIST
        gist = GIST(config, submodular_fn)
        gist.fit(df)
        gist_solution = gist.select()
        gist_value = gist._objective(gist_solution, data, dist_matrix)
        
        # Brute force optimal
        optimal_set, optimal_value = brute_force_optimal(
            data, dist_matrix, submodular_fn, config
        )
        
        approximation_ratio = gist_value / optimal_value
        assert approximation_ratio >= 0.5, \
            f"GIST achieved {approximation_ratio:.4f}, expected >= 0.5"
        
        print(f"Coverage utility test: ratio = {approximation_ratio:.4f}")
    
    def test_multiple_runs_convergence(self):
        """Run GIST multiple times to show convergence to > 0.5"""
        n, k = 30, 8
        ratios = []
        
        for seed in range(10):
            df = generate_submodular_breaking_dataset(n, seed=seed)
            data = df.values
            
            weights = np.random.RandomState(seed).rand(n)
            submodular_fn = LinearUtility(weights)
            
            config = GISTConfig(k=k, lambda_param=1.0, epsilon=0.05, distance_metric=DistanceMetric.EUCLIDEAN)
            dist_fn = DistanceMetric.EUCLIDEAN.get_function()
            dist_matrix = compute_pairwise_distances(data, dist_fn)
            
            gist = GIST(config, submodular_fn)
            gist.fit(df)
            gist_solution = gist.select()
            gist_value = gist._objective(gist_solution, data, dist_matrix)
            
            optimal_set, optimal_value = brute_force_optimal(
                data, dist_matrix, submodular_fn, config
            )
            
            ratio = gist_value / optimal_value
            ratios.append(ratio)
        
        avg_ratio = np.mean(ratios)
        min_ratio = np.min(ratios)
        
        assert min_ratio >= 0.5, \
            f"Minimum ratio {min_ratio:.4f} < 0.5"
        assert avg_ratio > 0.5, \
            f"Average ratio {avg_ratio:.4f} <= 0.5"
        
        print(f"Multiple runs: min={min_ratio:.4f}, avg={avg_ratio:.4f}")
    
    def test_different_lambda_values(self):
        """Test with different λ values"""
        n, k = 25, 7
        df = generate_test_dataset(n, d=8, seed=456)
        data = df.values
        
        np.random.seed(456)
        weights = np.random.rand(n)
        submodular_fn = LinearUtility(weights)
        dist_fn = DistanceMetric.EUCLIDEAN.get_function()
        dist_matrix = compute_pairwise_distances(data, dist_fn)
        
        for lambda_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
            config = GISTConfig(k=k, lambda_param=lambda_val, epsilon=0.05, distance_metric=DistanceMetric.EUCLIDEAN)
            
            gist = GIST(config, submodular_fn)
            gist.fit(df)
            gist_solution = gist.select()
            gist_value = gist._objective(gist_solution, data, dist_matrix)
            
            optimal_set, optimal_value = brute_force_optimal(
                data, dist_matrix, submodular_fn, config
            )
            
            ratio = gist_value / optimal_value
            assert ratio >= 0.5, \
                f"λ={lambda_val}: ratio {ratio:.4f} < 0.5"
            
            print(f"λ={lambda_val}: ratio = {ratio:.4f}")
