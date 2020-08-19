import numpy as np

#TODO generate triangle according to the vertices
def generate_triangle(vertices):
    triangles = []
    for i in range(vertices.shape[0]-1):
        for j in range(vertices.shape[1]-1):
            triangles.append([[i,j],[i+1,j],[i,j+1]])
            triangles.append([[i+1,j],[i+1,j+1],[i,j+1]])
    triangles = np.array(triangles,dtype=np.int)
    return triangles