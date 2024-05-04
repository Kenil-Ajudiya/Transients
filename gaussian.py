import numpy as np

# generates n-d gaussian. std and shape must be same length
def Gaussian(std, shape):
    std = np.array(std, ndmin=1)
    shape = np.array(shape, ndmin=1, dtype=float)
    
    axes = []
    for dim in range(shape.size):
        axes += [np.linspace(-shape[dim] / 2, shape[dim] / 2, int(shape[dim]))]
    
    coords = np.array(np.meshgrid(*axes, indexing='ij'))
    std_shape = np.ones(len(coords.shape), dtype=int)
    std_shape[0] = std.size
    std = np.reshape(std, std_shape)

    return np.exp(-np.sum((coords / std)**2, axis=0) / 2)