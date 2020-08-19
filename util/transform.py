import numpy as np


def transform(points,H):
    # points = points.reshape(-1,2)
    new_points = np.pad(points,((0,0),(0,1)),"constant",constant_values=1)
    new_points = new_points.T
    transformed_points = H.dot(new_points)
    transformed_points = transformed_points.T
    for i in range(transformed_points.shape[0]):
        transformed_points[i] = transformed_points[i]/transformed_points[i,2]
    new_points = transformed_points[:,:2]
    new_points = new_points.astype(np.int)
    return new_points