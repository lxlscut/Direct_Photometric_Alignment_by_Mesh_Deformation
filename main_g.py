import cv2
import numpy as np
import copy
from feature.match import Match
from mesh.bilinear_interpolation import bin_interpolate
from mesh.get_triangle import generate_triangle
from mesh.mesh import get_mesh_boxes, get_sample_point
from util.blending_average import blending_average
from util.draw import Draw
from util.gradient import calclulate_gradient, calculate_gradient_graph, gradient_gradient
from util.image_info import Image_info
from util.transform import transform
from util.triangle_similar import get_triangle_coefficient
import optimization
import texture_mapping


if __name__ == '__main__':
    src = cv2.imread("image/DSC00318.JPG")
    dst = cv2.imread("image/DSC00319.JPG")
    cv2.imshow("src",src)
    cv2.imshow("dst",dst)

    match = Match(src, dst)
    match.getInitialFeaturePairs()
    src_point = match.src_match
    dst_point = match.dst_match

    draw = Draw()
    H,no = cv2.findHomography(src_point,dst_point)
    img_info = Image_info()
    img_info.get_final_size(src,dst,H)

    canvas = np.zeros([img_info.height,img_info.width,1],dtype=np.uint8)
    canvas[img_info.offset_y:img_info.offset_y+dst.shape[0],img_info.offset_x:img_info.offset_x+dst.shape[1],:] = 255
    dst_warp = np.zeros([img_info.height,img_info.width,3],dtype=np.uint8)
    dst_warp[img_info.offset_y:img_info.offset_y+dst.shape[0],img_info.offset_x:img_info.offset_x+dst.shape[1],:] = dst
    # todo get mesh grid on src image
    mesh_boxes_src,src_x_num,src_y_num = get_mesh_boxes(src)
    mesh_boxes_src_show = mesh_boxes_src.reshape(-1,2)
    mesh_boxes_src_pic = draw.draw(src,mesh_boxes_src_show)

    #todo sample the point in source image
    sample_point_src = get_sample_point(src)
    sample_point_src = sample_point_src.reshape(-1,2)
    tran_sample_points = transform(sample_point_src,H)

    #todo get the sample point in overlap ragion
    sample_point_src_or = []
    tran_sample_points_or = []
    for i in range(tran_sample_points.shape[0]):
        if 0<tran_sample_points[i,0]+img_info.offset_x<canvas.shape[1] and 0<tran_sample_points[i,1]+img_info.offset_y<canvas.shape[0]:
            if canvas[tran_sample_points[i,1],tran_sample_points[i,0]]==255:
                sample_point_src_or.append(sample_point_src[i])
                tran_sample_points_or.append(tran_sample_points[i])
    sample_point_src_or = np.array(sample_point_src_or)
    tran_sample_points_or = np.array(tran_sample_points_or)

    #todo bilinear inpolation sample point with meshgrid
    weight,location = bin_interpolate(sample_point_src_or,mesh_boxes_src)
    weight = np.squeeze(weight)

    #todo we transfrom the image to gray the use the value of pixel as intensity
    src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    dst_warp_gray = cv2.cvtColor(dst_warp,cv2.COLOR_BGR2GRAY)

    #todo caculate the gradient at sample point of source image
    #todo processs the first condition
    #todo for every match sample point
    grad = calclulate_gradient(sample_point_src_or,src_gray)
    bs = []
    for i in range(sample_point_src_or.shape[0]):
        # intensity difference
        b1 = -int(dst_warp_gray[tran_sample_points_or[i,1],tran_sample_points_or[i,0]])+int(src_gray[sample_point_src_or[i,1],sample_point_src_or[i,0]])
        b2 = grad[i,0]*sample_point_src_or[i,0]+grad[i,1]*sample_point_src_or[i,1]
        b = b2-b1
        bs.append(b)
    bs = np.array(bs,dtype=np.float32)
    # thrid calculate every cofficient
    cofficients = []
    print(weight.shape)
    for j in range(sample_point_src_or.shape[0]):
        for k in range(weight[j].shape[0]):
            cofficient = [weight[j,k]*grad[j,0],weight[j,k]*grad[j,1]]
            cofficients.append(cofficient)
    cofficients = np.array(cofficients,dtype=np.float32)
    cofficients = np.squeeze(cofficients)
    cofficients = cofficients.reshape(-1,4,2)


    #TODO process second condition(constrain)
    # to every point not in overlap ragion,we let them have a similarity transform
    triangles = generate_triangle(vertices=mesh_boxes_src)
    triangle_coefficient = get_triangle_coefficient(vertices=mesh_boxes_src,triangle=triangles)


    # TODO process the thrid constrain,the gradient constrain
    src_warp_gradient = calculate_gradient_graph(src_gray)
    dst_warp_gradient = calculate_gradient_graph(dst_warp_gray)
    grad2 = gradient_gradient(sample_point_src_or,src_warp_gradient)
    ###########################################################################
    ############ E = ∑||Gt(pi+τ(pi))−Gs(pi)||^2################################
    ###########################################################################
    cofficients_g = []
    bbss = []
    for i in range(sample_point_src_or.shape[0]):
        bb = grad2[i]*sample_point_src_or[i] + src_warp_gradient[sample_point_src_or[i,1],sample_point_src_or[i,0]] - dst_warp_gradient[sample_point_src_or[i,1],sample_point_src_or[i,0]]
        bbss.append(bb)
        for w in weight[i]:
            cofficient_g = [w*grad2[i,0],w*grad2[i,1]]
            cofficients_g.append(cofficient_g)
    cofficients_g = np.array(cofficients_g)
    cofficients_g = cofficients_g.reshape(-1,4,2)
    bbss = np.array(bbss)




    print(">>>>>>>>>>>>>>>>>>>>>>>>>")
    print(triangles.shape)
    print(triangle_coefficient.shape)
    print(cofficients.shape)
    print(location.shape)
    print(bs.shape)
    print(mesh_boxes_src.shape)
    print(cofficients_g.shape)
    print(bbss.shape)
    # triangles = triangles * 0
    # triangle_coefficient = triangle_coefficient * 0
    # cofficients = cofficients * 0
    # location = location * 0
    # bs = bs * 0
    # mesh_boxes_src = mesh_boxes_src * 0
    # cofficients_g = cofficients_g * 0
    # bbss = bbss * 0

    c = optimization.optimize(triangles,triangle_coefficient,cofficients,location,bs,mesh_boxes_src,cofficients_g,bbss,0.2)
    c = c.astype(np.int)
    c = c.reshape(src_y_num,src_x_num,2)

    """the offset of texture_mapping bring"""
    offset_x = abs(min(np.min(c[:,:,0]),0))
    offset_y = abs(min(np.min(c[:,:,1]),0))


    # todo get the image after texture_mapping
    final_result = texture_mapping.texture_mapping(mesh_boxes_src.astype(np.int), c.astype(np.int),
                                    src)
    final_result = final_result.astype(np.uint8)
    cv2.imshow("final_result",final_result)

    # todo new image info
    img_info_2 = Image_info()
    img_info_2.get_final_size(src,dst,H)

    #todo warping the result
    src_warp = cv2.warpPerspective(final_result, H, (img_info.width, img_info.height))
    dst_warp = np.zeros_like(src_warp)
    dst_warp[img_info_2.offset_y:dst.shape[0] + img_info_2.offset_y, img_info_2.offset_x:dst.shape[1] + img_info_2.offset_x,
    :] = dst[:, :, :]
    result,mask = blending_average(src_warp,dst_warp)
    # cv2.imshow("mask",mask*255)



    #todo
    cv2.imshow("dst_warp",result)
    cv2.imshow("mesh_boxes_src_pic",mesh_boxes_src_pic)
    cv2.imshow("canvas",canvas)

    cv2.waitKey(0)

