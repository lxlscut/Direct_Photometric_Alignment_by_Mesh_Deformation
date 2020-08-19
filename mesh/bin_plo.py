import math
import numpy as np


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def bilinear_interpolation(p, a, b, c, d):
    """
    :param a:
    :param point: feature point
    :param vertices: the vertices of the mesh surround the point,Clockwise
    :return: the bilinear interpolation coefficient
    """
    #     todo 公式纸上推导
    #     todo for u ,aau**2+bb*u+cc = 0
    alpha = np.array([[b[0] - a[0]], [b[1] - a[1]]])
    beta = np.array([[d[0] - a[0]], [d[1] - a[1]]])
    theta = np.array([[a[0] - b[0] + c[0] - d[0]], [a[1] - b[1] + c[1] - d[1]]])
    gama = np.array([[a[0] - p[0]], [a[1] - p[1]]])

    aa = alpha[1] * theta[0] - theta[1] * alpha[0]
    bb = alpha[1] * beta[0] - beta[1] * alpha[0] + gama[1] * theta[0] - gama[0] * theta[1]
    cc = beta[0] * gama[1] - beta[1] * gama[0]
    global u
    if aa == 0:
        u = -cc / bb
    elif (bb ** 2 - 4 * aa * cc) < 0:
        print("no answer for the equation")
    else:
        u1 = (-bb + math.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)
        u2 = (-bb - math.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)
        if 0 <= u1 <= 1:
            u = u1
        else:
            u = u2
    global v
    aaa = theta[0] * beta[1] - beta[0] * theta[1]
    bbb = gama[1] * theta[0] - theta[1] * gama[0] + alpha[0] * beta[1] - beta[0] * alpha[1]
    ccc = gama[1] * alpha[0] - alpha[1] * gama[0]
    if aaa == 0:
        v = -ccc / bbb
    elif (bbb ** 2 - 4 * aaa * ccc) < 0:
        print("no answer for the equation")
    else:
        v1 = (-bbb + math.sqrt(bbb ** 2 - 4 * aaa * ccc)) / (2 * aaa)
        v2 = (-bb - math.sqrt(bbb ** 2 - 4 * aaa * ccc)) / (2 * aaa)
        if 0 <= v1 <= 1:
            v = v1
        else:
            v = v2
    # print(u, v)
    # test(p, a, b, c, d, u, v)
    # print(1 - u - v + u * v,u - u * v,u * v,v - u * v)
    return [1 - u - v + u * v,u - u * v,u * v,v - u * v]





def test(p, a, b, c, d, u, v):
    x = a*(1 - u - v + u * v) + b*(u - u * v) + c*(u * v)+d*(v - u * v)
    err = (p - x)
    err = err[0] ** 2 + err[1] ** 2
    print(err)
    if err < 0.1:
        print("验证通过")


if __name__ == '__main__':
    p = np.array([1, 1]).astype(np.float)
    a = np.array([0, 0]).astype(np.float)
    b = np.array([2, 0]).astype(np.float)
    c = np.array([2, 2]).astype(np.float)
    d = np.array([0, 2]).astype(np.float)
    u, v = bilinear_interpolation(p, a, b, c, d)
    test(p, a, b, c, d, u, v)
    print(u, v)