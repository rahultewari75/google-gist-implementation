# GIST: Greedy Independent Set Thresholding

My implementation of the GIST algorithm for **Max-Min Diversification with Monotone Submodular Utility** from the paper:

> **GIST: Greedy Independent Set Thresholding for Max-Min Diversification with Submodular Utility**  
> [arXiv:2405.18754](https://arxiv.org/pdf/2405.18754)

## Overview

GIST solves the MDMS problem: given a set of points in a metric space, select a subset $S$ of size at most $k$ that maximizes:

$$f(S) = g(S) + \lambda \cdot \text{div}(S)$$

where:
- $g(S)$ is a **monotone submodular** utility function
- $\text{div}(S) = \min_{u,v \in S, u \neq v} \text{dist}(u,v)$ is the **max-min diversity**
- $\lambda \geq 0$ controls the trade-off between utility and diversity

**Approximation Guarantee**: GIST achieves atleast a **1/2-approximation** for general submodular utilities 

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
import pandas as pd
from gist import GIST, GISTConfig, DistanceMetric, LinearUtility

# Generate or load your data
df = pd.DataFrame(np.random.randn(100, 10))  # 100 points, 10 features

# Create a submodular utility function
weights = np.random.rand(100)
submodular_fn = LinearUtility(weights)

# Configure GIST
config = GISTConfig(
    k=20,                              # Select at most 20 points
    lambda_param=1.0,                  # Diversity strength
    epsilon=0.05,                      # Error parameter
    distance_metric=DistanceMetric.EUCLIDEAN  # Distance metric enum
)

# Run GIST
gist = GIST(config, submodular_fn)
gist.fit(df)
selected_indices = gist.select()

# Get selected data
selected_df = gist.get_selected_data(df)
```

## Project Structure

```
subset_selection_impl/
├── gist/
│   ├── algorithm.py          # Main GIST algorithm
│   ├── submodular.py         # Submodular utility functions
│   ├── distance.py          # Distance metrics
│   └── types.py             # Type definitions
├── tests/
│   ├── test_gist.py         # Unit tests
│   ├── test_approximation.py # Approximation guarantee tests
│   └── test_utils.py        # Test utilities
├── examples/
│   └── example_usage.py
├── requirements.txt
└── README.md
```

## Features

### Submodular Utility Functions

- **LinearUtility**: $g(S) = \sum_{v \in S} w(v)$
- **CoverageUtility**: $g(S) = \sum_{i} \max_{j \in S} c_{ij}$

### Distance Metrics

- `DistanceMetric.EUCLIDEAN` - Euclidean distance (L2)
- `DistanceMetric.COSINE` - Cosine distance

Use the `DistanceMetric` enum for type-safe metric selection.

## Algorithm

GIST works by:

1. **Initialization**: Start with greedy solution (distance threshold $d=0$)
2. **Diametrical Pair**: Consider two points with maximum distance
3. **Threshold Sweeping**: Iterate over distance thresholds $d \in D$ where:
   $$D = \{(1+\varepsilon)^i \cdot \varepsilon \cdot d_{\max}/2 : (1+\varepsilon)^i \leq 2/\varepsilon\}$$
4. **GreedyIndependentSet**: For each threshold, build a maximal independent set greedily
5. **Return Best**: Return the solution with highest objective value

## Testing

Run the test suite to validate the 1/2-approximation guarantee against brute force optimal:

```bash
pytest tests/
```


## Example Output

```
Selected 20 points
Indices: [2, 5, 12, 18, 23, 31, 34, 41, 47, 52, 58, 63, 69, 74, 80, 85, 91, 94, 96, 99]
Objective score: 15.2341

Selected data shape: (20, 10)
```

## References

- [GIST Paper (arXiv)](https://arxiv.org/pdf/2405.18754)
- [Nemhauser et al. (1978) - Submodular Maximization](http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2019/Paperssubmodular/nemhauser.pdf)
