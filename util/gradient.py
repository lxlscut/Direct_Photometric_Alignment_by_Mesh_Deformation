import numpy as np
import cv2
from numba import jit
#TODO calculate the gradient of a point in the image
def cal_gradient(point,img):
    """
    :param point: the point to be calculated
    :param img: the gray image
    :return: the gradient of vertical and horizontal
    """
    if 0<point[0]<img.shape[1]-1 and 0<point[1]<img.shape[0]-1:
        grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]-1]))/2
        grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1]-1,point[0]]))/2
        # print("dassd" + str(grad_x) + " " +  str(grad_y) + " " + str(img[point[1],point[0]+1]))
    elif point[0]==0 and 0<point[1]<img.shape[0]-1:
        grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]]))/2
        grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1]-1,point[0]]))/2
    elif point[0]==img.shape[1]-1 and 0<point[1]<img.shape[0]-1:
        grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
        grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1]-1,point[0]]))/2

    elif 0<point[0]<img.shape[1]-1 and point[1]==0:
        grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
        grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1],point[0]]))/2
    elif 0<point[0]<img.shape[1]-1 and point[1]==img.shape[0]-1:
        grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
        grad_y = (int(img[point[1],point[0]])-int(img[point[1]-1,point[0]]))/2

    elif point[0]==0 and point[1]==0:
        grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]]))/2
        grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1],point[0]]))/2
    elif point[0]==img.shape[1]-1 and point[1]==img.shape[0]-1:
        grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
        grad_y = (int(img[point[1],point[0]])-int(img[point[1]-1,point[0]]))/2
    elif point[0]==0 and point[1]==img.shape[0]-1:
        grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]]))/2
        grad_y = (int(img[point[1],point[0]])-int(img[point[1]-1,point[0]]))/2
    elif point[0]==img.shape[1]-1 and point[1]==0:
        grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
        grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1],point[0]]))/2


    # if mask[point[1],point[0]-1]>0 and mask[point[1],point[0]+1]>0 and mask[point[1]-1,point[0]]>0 and mask[point[1]+1,point[0]]>0:
    #     grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]-1]))/2
    #     grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1]-1,point[0]]))/2
    # elif mask[point[1],point[0]-1]>0 and mask[point[1],point[0]+1]==0 and mask[point[1]-1,point[0]]>0 and mask[point[1]+1,point[0]]>0:
    #     grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
    #     grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1]-1,point[0]]))/2
    # elif mask[point[1],point[0]-1]==0 and mask[point[1],point[0]+1]>0 and mask[point[1]-1,point[0]]>0 and mask[point[1]+1,point[0]]>0:
    #     grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]]))/2
    #     grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1]-1,point[0]]))/2
    # elif mask[point[1],point[0]-1]>0 and mask[point[1],point[0]+1]>0 and mask[point[1]-1,point[0]]==0 and mask[point[1]+1,point[0]]>0:
    #     grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]-1]))/2
    #     grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1],point[0]]))/2
    # elif mask[point[1],point[0]-1]>0 and mask[point[1],point[0]+1]>0 and mask[point[1]-1,point[0]]>0 and mask[point[1]+1,point[0]]==0:
    #     grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]-1]))/2
    #     grad_y = (int(img[point[1],point[0]])-int(img[point[1]-1,point[0]]))/2
    # elif mask[point[1],point[0]-1]==0 and mask[point[1],point[0]+1]>0 and mask[point[1]-1,point[0]]==0 and mask[point[1]+1,point[0]]>0:
    #     grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]]))/2
    #     grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1],point[0]]))/2
    # elif mask[point[1],point[0]-1]>0 and mask[point[1],point[0]+1]==0 and mask[point[1]-1,point[0]]>0 and mask[point[1]+1,point[0]]==0:
    #     grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
    #     grad_y = (int(img[point[1],point[0]])-int(img[point[1]-1,point[0]]))/2
    # elif mask[point[1],point[0]-1]==0 and mask[point[1],point[0]+1]>0 and mask[point[1]-1,point[0]]>0 and mask[point[1]+1,point[0]]==0:
    #     grad_x = (int(img[point[1],point[0]+1])-int(img[point[1],point[0]]))/2
    #     grad_y = (int(img[point[1],point[0]])-int(img[point[1]-1,point[0]]))/2
    # elif mask[point[1],point[0]-1]>0 and mask[point[1],point[0]+1]==0 and mask[point[1]-1,point[0]]==0 and mask[point[1]+1,point[0]]>0:
    #     grad_x = (int(img[point[1],point[0]])-int(img[point[1],point[0]-1]))/2
    #     grad_y = (int(img[point[1]+1,point[0]])-int(img[point[1],point[0]]))/2


    return [grad_x,grad_y]


def calclulate_gradient(points,img):
    """
    :param points: the set of point which need to calculate its gradient
    :param img: the correspond image
    :return: the numpy array of [grad_x,grad_y]
    """
    gradients = []
    print("points shape" + str(points.shape))
    for i in range(points.shape[0]):
        gradient = cal_gradient(points[i],img)
        gradients.append(gradient)
    gradients = np.array(gradients,dtype=np.float32)
    return gradients

# @jit(nopython=True)
def calculate_gradient_graph(img1):
    """
    :param img: the gray style image
    :return: the gradient of the image
    """
    # img1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gradient = np.zeros((img1.shape[0],img1.shape[1],2),dtype=np.float32)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if 0<i<img1.shape[0]-1 and 0<j<img1.shape[1]-1:
                gradient[i,j,0] = (int(img1[i,j+1]) - int(img1[i,j-1]))/2
                gradient[i,j,1] = (int(img1[i+1,j]) - int(img1[i-1,j]))/2
                # print(gradient[i,j,0],gradient[i,j,1])
            elif i==0 and 0<j<img1.shape[1]-1:
                gradient[i,j,0] = (int(img1[i,j+1]) - int(img1[i,j-1]))/2
                gradient[i,j,1] = (int(img1[i+1,j]) - int(img1[i,j]))/2
            elif i==img1.shape[0]-1 and 0<j<img1.shape[1]-1:
                gradient[i,j,0] = (int(img1[i,j+1]) - int(img1[i,j-1]))/2
                gradient[i,j,1] = (int(img1[i,j]) - int(img1[i-1,j]))/2
            elif 0<i<img1.shape[0]-1 and j==0:
                gradient[i,j,0] = (int(img1[i,j+1]) - int(img1[i,j]))/2
                gradient[i,j,1] = (int(img1[i+1,j]) - int(img1[i-1,j]))/2
            elif 0 < i < img1.shape[0] - 1 and j == img1.shape[1]-1:
                gradient[i,j,0] = (int(img1[i,j]) - int(img1[i,j-1]))/2
                gradient[i,j,1] = (int(img1[i+1,j]) - int(img1[i-1,j]))/2
            elif i==0 and j == 0:
                gradient[i,j,0] = (int(img1[i,j+1]) - int(img1[i,j]))/2
                gradient[i,j,1] = (int(img1[i+1,j]) - int(img1[i,j]))/2
            elif i==img1.shape[0]-1 and j==img1.shape[1]-1:
                gradient[i,j,0] = (int(img1[i,j]) - int(img1[i,j]))/2
                gradient[i,j,1] = (int(img1[i,j]) - int(img1[i,j]))/2
            elif i==0 and j==img1.shape[1]-1:
                gradient[i,j,0] = (int(img1[i,j]) - int(img1[i,j-1]))/2
                gradient[i,j,1] = (int(img1[i+1,j]) - int(img1[i,j]))/2
            elif i==img1.shape[0]-1 and j==0:
                gradient[i,j,0] = (int(img1[i,j+1]) - int(img1[i,j]))/2
                gradient[i,j,1] = (int(img1[i,j]) - int(img1[i-1,j]))/2
    return gradient


def gradient_gradient(points,gradient):
    """
    :param point: optical flow sample point
    :param gradient: gradient_graph,first channel is x，second channel is y
    :return:
    """
    grad_g = []
    for point in points:
        # if mask[point[1], point[0] - 1] > 0 and mask[point[1], point[0] + 1] > 0 and mask[
        #     point[1] - 1, point[0]] > 0 and mask[point[1] + 1, point[0]] > 0:
        #     grad_x = (gradient[point[1],point[0]+1,0]-gradient[point[1],point[0]-1,0])/2
        #     grad_y = (gradient[point[1]+1,point[0],1]-gradient[point[1]-1,point[0],1])/2
        #
        # elif mask[point[1], point[0] - 1] > 0 and mask[point[1], point[0] + 1] == 0 and mask[
        #     point[1] - 1, point[0]] > 0 and mask[point[1] + 1, point[0]] > 0:
        #     grad_x = (gradient[point[1], point[0], 0] - gradient[point[1], point[0]-1 , 0]) / 2
        #     grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1] - 1, point[0], 1]) / 2
        #
        # elif mask[point[1], point[0] - 1] == 0 and mask[point[1], point[0] + 1] > 0 and mask[
        #     point[1] - 1, point[0]] > 0 and mask[point[1] + 1, point[0]] > 0:
        #     grad_x = (gradient[point[1], point[0]+1 , 0] - gradient[point[1], point[0] , 0]) / 2
        #     grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1] - 1, point[0], 1]) / 2
        #
        # elif mask[point[1], point[0] - 1] > 0 and mask[point[1], point[0] + 1] > 0 and mask[
        #     point[1] - 1, point[0]] == 0 and mask[point[1] + 1, point[0]] > 0:
        #     grad_x = (gradient[point[1], point[0] + 1, 0] - gradient[point[1], point[0] - 1, 0]) / 2
        #     grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1], point[0], 1]) / 2
        #
        # elif mask[point[1], point[0] - 1] > 0 and mask[point[1], point[0] + 1] > 0 and mask[
        #     point[1] - 1, point[0]] > 0 and mask[point[1] + 1, point[0]] == 0:
        #     grad_x = (gradient[point[1],point[0]+1,0]-gradient[point[1],point[0]-1,0])/2
        #     grad_y = (gradient[point[1],point[0],1]-gradient[point[1]-1,point[0],1])/2
        #
        # elif mask[point[1], point[0] - 1] == 0 and mask[point[1], point[0] + 1] > 0 and mask[
        #     point[1] - 1, point[0]] == 0 and mask[point[1] + 1, point[0]] > 0:
        #     grad_x = (gradient[point[1],point[0]+1,0]-gradient[point[1],point[0],0])/2
        #     grad_y = (gradient[point[1]+1,point[0],1]-gradient[point[1],point[0],1])/2
        #
        # elif mask[point[1], point[0] - 1] > 0 and mask[point[1], point[0] + 1] == 0 and mask[
        #     point[1] - 1, point[0]] > 0 and mask[point[1] + 1, point[0]] == 0:
        #     grad_x = (gradient[point[1],point[0],0]-gradient[point[1],point[0]-1,0])/2
        #     grad_y = (gradient[point[1],point[0],1]-gradient[point[1]-1,point[0],1])/2
        #
        # elif mask[point[1], point[0] - 1] == 0 and mask[point[1], point[0] + 1] > 0 and mask[
        #     point[1] - 1, point[0]] > 0 and mask[point[1] + 1, point[0]] == 0:
        #     grad_x = (gradient[point[1],point[0]+1,0]-gradient[point[1],point[0],0])/2
        #     grad_y = (gradient[point[1],point[0],1]-gradient[point[1]-1,point[0],1])/2
        #
        # elif mask[point[1], point[0] - 1] > 0 and mask[point[1], point[0] + 1] == 0 and mask[
        #     point[1] - 1, point[0]] == 0 and mask[point[1] + 1, point[0]] > 0:
        #     grad_x = (gradient[point[1],point[0],0]-gradient[point[1],point[0]-1,0])/2
        #     grad_y = (gradient[point[1]+1,point[0],1]-gradient[point[1],point[0],1])/2

        if 0<point[0]<gradient.shape[1]-1 and 0<point[1]<gradient.shape[0]-1:
            grad_x = (gradient[point[1],point[0]+1,0]-gradient[point[1],point[0]-1,0])/2
            grad_y = (gradient[point[1]+1,point[0],1]-gradient[point[1]-1,point[0],1])/2

        elif  point[0] == 0 and 0 < point[1] < gradient.shape[0] - 1:
            grad_x = (gradient[point[1], point[0] + 1, 0] - gradient[point[1], point[0] , 0]) / 2
            grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1] - 1, point[0], 1]) / 2

        elif point[0] == gradient.shape[1] - 1 and 0 < point[1] < gradient.shape[0] - 1:
            grad_x = (gradient[point[1], point[0] , 0] - gradient[point[1], point[0] - 1, 0]) / 2
            grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1] - 1, point[0], 1]) / 2

        elif 0 < point[0] < gradient.shape[1] - 1 and point[1] ==0:
            grad_x = (gradient[point[1], point[0] + 1, 0] - gradient[point[1], point[0] - 1, 0]) / 2
            grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1], point[0], 1]) / 2
        elif 0 < point[0] < gradient.shape[1] - 1 and  point[1] == gradient.shape[0] - 1:
            grad_x = (gradient[point[1], point[0] + 1, 0] - gradient[point[1], point[0] - 1, 0]) / 2
            grad_y = (gradient[point[1], point[0], 1] - gradient[point[1] - 1, point[0], 1]) / 2


        elif point[0] ==0 and point[1] == 0:
            grad_x = (gradient[point[1], point[0] + 1, 0] - gradient[point[1], point[0], 0]) / 2
            grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1], point[0], 1]) / 2

        elif point[0] == gradient.shape[1] - 1 and point[1] == gradient.shape[0] - 1:
            grad_x = (gradient[point[1], point[0], 0] - gradient[point[1], point[0] - 1, 0]) / 2
            grad_y = (gradient[point[1], point[0], 1] - gradient[point[1] - 1, point[0], 1]) / 2


        elif point[0] ==0 and point[1] == gradient.shape[0]:
            grad_x = (gradient[point[1], point[0] + 1, 0] - gradient[point[1], point[0], 0]) / 2
            grad_y = (gradient[point[1], point[0], 1] - gradient[point[1] - 1, point[0], 1]) / 2
        elif point[0] == gradient.shape[1] - 1 and  point[1] == 0:
            grad_x = (gradient[point[1], point[0], 0] - gradient[point[1], point[0] - 1, 0]) / 2
            grad_y = (gradient[point[1] + 1, point[0], 1] - gradient[point[1], point[0], 1]) / 2

        grad_g.append([grad_x,grad_y])
    grad_g = np.array(grad_g)
    return grad_g