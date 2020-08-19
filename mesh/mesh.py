import numpy as np

from util.get_param import get_param


def get_mesh_boxes(img):
    width = get_param("mesh_width")
    height = get_param("mesh_height")
    x_no = int(img.shape[1]/width)
    y_no = int(img.shape[0]/height)

    x = np.linspace(0,img.shape[1]-1,x_no)
    y = np.linspace(0,img.shape[0]-1,y_no)

    # x = range(0,img.shape[1]-1,50)
    # y = range(0,img.shape[0]-1,50)

    [x_d,y_d] = np.meshgrid(x,y)
    vertices = np.stack([x_d,y_d],axis=-1)
    return vertices.astype(np.int),x_no,y_no


def get_sample_point(img):

    width = get_param("sample_width")
    height = get_param("sample_height")
    x_no = int(img.shape[1]/width)
    y_no = int(img.shape[0]/height)

    x = np.linspace(0, img.shape[1] - 1, x_no)
    y = np.linspace(0, img.shape[0] - 1, y_no)
    [x_d, y_d] = np.meshgrid(x, y)
    vertices = np.stack([x_d, y_d], axis=-1)
    return vertices.astype(np.int)

