import numpy as np

# TODO The bilinear_interpolation should calculate 2 result:
#  1.the weight of 4 vertices. 2.the location of 4 vertices(or no)
from mesh.bin_plo import bilinear_interpolation


def bin_interpolate(sample_points,vertices):
    cofficient = []
    location = []
    sample_points = sample_points.reshape(-1,2)
    x_measure = np.array([i for i in vertices[0,:,0]])
    y_measure = np.array([i for i in vertices[:,0,1]])
    print(x_measure)
    print(y_measure)
    # vertices = vertices.reshape(-1,2)
    for i in range(sample_points.shape[0]):
        # here the value is 50 cause we already know that the value in mesh.py is also 50. in the end we will replace the value with the value in the configuration file

        for j in range(y_measure.shape[0]):
            if sample_points[i,1]<y_measure[j]:
                # print(sample_points[i, 1], x_measure[j])
                y_no=j-1
                break
        for k in range(x_measure.shape[0]):
            if sample_points[i,0]<x_measure[k]:
                # print(sample_points[i,0],x_measure[k])
                x_no=k-1
                break

        # print(vertices.shape,x_no,y_no)
        vertice = np.array([vertices[y_no,x_no,:],vertices[y_no,x_no+1,:],vertices[y_no+1,x_no+1,:],vertices[y_no+1,x_no,:]])
        cof = bilinear_interpolation(sample_points[i],vertice[0],vertice[1],vertice[2],vertice[3])
        cofficient.append(cof)
        location.append([y_no,x_no])
    cofficient = np.array(cofficient,dtype=np.float32)
    location = np.array(location,dtype=np.int)

    return cofficient,location




