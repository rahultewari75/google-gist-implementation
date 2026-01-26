import numpy as np
import pandas as pd
from gist.algorithm import GIST
from gist.types import GISTConfig, DistanceMetric
from gist.submodular import LinearUtility

def main():
    # Generate sample data
    n, d = 100, 10
    np.random.seed(42)
    data = np.random.randn(n, d)
    df = pd.DataFrame(data, columns=[f'feat_{i}' for i in range(d)])
    
    # Create submodular utility function
    weights = np.random.rand(n)
    submodular_fn = LinearUtility(weights)
    
    # Configure GIST
    config = GISTConfig(
        k=20,
        lambda_param=1.0,
        epsilon=0.05,
        distance_metric=DistanceMetric.EUCLIDEAN
    )
    
    # Run GIST
    gist = GIST(config, submodular_fn)
    gist.fit(df)
    selected_indices = gist.select()
    
    # Compute objective score
    score = gist._objective(selected_indices, gist.data, gist.dist_matrix)
    
    print(f"Selected {len(selected_indices)} points")
    print(f"Indices: {sorted(selected_indices)}")
    print(f"Objective score: {score:.4f}")
    
    # Get selected data
    selected_df = gist.get_selected_data(df)
    print(f"\nSelected data shape: {selected_df.shape}")

if __name__ == '__main__':
    main()
