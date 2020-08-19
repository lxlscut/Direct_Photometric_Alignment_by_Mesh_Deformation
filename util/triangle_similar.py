import numpy as np


def get_triangle_coefficient(triangle, vertices):
    triangle_coefficient = np.zeros((triangle.shape[0], 2), dtype=np.float64)
    for i in range(triangle.shape[0]):
        a = vertices[triangle[i, 0, 0], triangle[i, 0, 1]]
        b = vertices[triangle[i, 1, 0], triangle[i, 1, 1]]
        c = vertices[triangle[i, 2, 0], triangle[i, 2, 1]]


        u = ((a[1]-b[1])*(c[1]-b[1]) - (a[0]-b[0])*(b[0]-c[0]))/((c[1]-b[1])**2+(c[0]-b[0])**2)
        v = (a[0]-b[0]-u*(c[0]-b[0]))/(c[1]-b[1])
        triangle_coefficient[i] = [u, v]
        # test_uv(a, b, c, u, v)
    return triangle_coefficient


def test_uv(a, b, c, u, v):
    s = np.array([[0, 1], [-1, 0]], dtype=np.float)
    result = a - (b + u * (c - b) + v * s.dot(c - b))
    value = result[0] ** 2 + result[1] ** 2
    print(value)
    if value < 0.1:
        print("u,v验证通过")
