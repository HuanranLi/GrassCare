# GrassCare

GrassCare is a Python package designed for visualizing high-dimensional Grassmannians on the Poincaré disk. This package provides tools to effectively project and preserve the geometry of the original high-dimensional Grassmannian in a 2D embedding. The core function of the package is `grasscare`, which utilizes gradient descent methods to minimize the KL-divergence between the affinity matrices of the Grassmannian and the Poincaré disk.

## Features

- **Efficient Projection**: GrassCare determines the optimal angle for projecting Grassmannians onto the Poincaré disk, preserving geodesic distances.
- **High-dimensional Support**: Maintains the geometry of high-dimensional Grassmannians while embedding them into a low-dimensional plane, effectively preventing distortion during dimension reduction.
- **Various Gradient Descent Methods**: Supports multiple gradient descent methods, including MomentumGD and ADAM, for optimizing the embedding.

## Installation

To use GrassCare, clone the repository and install the required dependencies:

```bash
git clone https://github.com/HuanranLi/GrassCare.git
cd GrassCare
pip install -r requirements.txt
```

## Usage

### Importing the Package

```python
import numpy as np
import matplotlib.pyplot as plt
from GrassCare import grasscare
```

### Core Function: `grasscare`

The `grasscare` function is the primary interface of this package. Below is an example of how to use this function:

```python
# Example usage
U_array = np.random.rand(10, 3, 3)  # Replace with actual Grassmannian data
history = grasscare(U_array, gradient_method='ADAM', eta=0.01, max_iter=500, init='random', verbose=1)

# Visualizing the embedding
final_embedding = history[-1]['b_array']
plt.scatter(final_embedding[:, 0], final_embedding[:, 1])
plt.show()
```

### Parameters

- `U_array`: Input array representing the Grassmannian data.
- `gradient_method`: The method for gradient descent (`'MomentumGD'` or `'ADAM'`).
- `eta`: Learning rate for the gradient descent.
- `moment`: Momentum factor (used only with `'MomentumGD'`).
- `max_iter`: Maximum number of iterations.
- `init`: Initialization method (`'random'` or `'3D'`).
- `beta`: Parameter controlling the spread in the Poincaré disk.
- `verbose`: If set to 1, prints the loss at each iteration.

### Example

An example of using `grasscare` can be found in the `main.py` script included in this repository.

## Paper

The detailed methodology and theoretical background of GrassCare can be found in the associated research paper: [GrassCaré: Visualizing the Grassmannian on the Poincaré Disk](https://link.springer.com/article/10.1007/s42979-023-02597-0).
