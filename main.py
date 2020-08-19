import cv2
import numpy as np
from feature.match import Match
from mesh.bilinear_interpolation import bin_interpolate
from mesh.get_triangle import generate_triangle
from mesh.mesh import get_mesh_boxes, get_sample_point
from util.blending_average import blending_average
from util.draw import Draw
from util.gradient import calclulate_gradient, calculate_gradient_graph, gradient_gradient
from util.image_info import Image_info
import optimization
import texture_mapping

from util.triangle_similar import get_triangle_coefficient

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

    # TODO stitching tow image by global homography, the fusion methed is feathering
    src_warp = cv2.warpPerspective(src,H,(img_info.width,img_info.height))
    dst_warp = np.zeros_like(src_warp)
    dst_warp[img_info.offset_y:dst.shape[0]+img_info.offset_y,img_info.offset_x:dst.shape[1]+img_info.offset_x,:] = dst[:,:,:]
    result,mask = blending_average(src_warp,dst_warp)
    cv2.imshow("mask",mask*255)




    #TODO optimize the result by optical flow

    #todo STEP1 todo first generate rugular mesh box
    mesh_boxes_src,src_x_num,src_dst_num = get_mesh_boxes(src_warp)
    mesh_boxes_dst,dst_x_num,dst_y_num = get_mesh_boxes(dst_warp)

    #todo STEP2 sample pixels from the img, according to the paper we sample the pixel with a interval of 3 pixel on both vertical and herizontal
    sample_vertices = get_sample_point(src_warp).reshape(-1,2)

    #todo get the sample pixel points in overlap ragion
    sample_vertices_or = []
    for sv in range(sample_vertices.shape[0]):
        if mask[sample_vertices[sv,1],sample_vertices[sv,0]]>0:
            sample_vertices_or.append(sample_vertices[sv])
    sample_vertices_or = np.array(sample_vertices_or,dtype=np.int)
    sample_vertices_or_pic = draw.draw(src_warp,sample_vertices_or)
    cv2.imshow("sample_vertices_or_pic",sample_vertices_or_pic)

    #todo Bilinear  interpolate the sample point in overlap ragion by the vertices, here we set dst_warp as the target image, src_warp as reference image
    weight,location = bin_interpolate(sample_vertices_or,mesh_boxes_src)
    weight = np.squeeze(weight)
    # print(mesh_boxes_src.shape)
    # for i in range(location.shape[0]):
    #     if  mask[mesh_boxes_src[location[i,0],location[i,1]][1],mesh_boxes_src[location[i,0],location[i,1]][0]]==0 or \
    #         mask[mesh_boxes_src[location[i,0]+1,location[i,1]+1][1],mesh_boxes_src[location[i,0]+1,location[i,1]+1][0]]==0 or \
    #         mask[mesh_boxes_src[location[i,0]+1,location[i,1]][1],mesh_boxes_src[location[i,0]+1,location[i,1]][0]]==0 or \
    #         mask[mesh_boxes_src[location[i, 0], location[i, 1]+1][1],mesh_boxes_src[location[i, 0], location[i, 1]+1][0]] == 0:
    #             np.delete(weight,i,axis=0)
    #             np.delete(location,i,axis=0)
    #             np.delete(sample_vertices_or,i,axis=0)

    #todo we transfrom the image to gray the use the value of pixel as intensity
    src_warp_gray = cv2.cvtColor(src_warp,cv2.COLOR_BGR2GRAY)
    dst_warp_gray = cv2.cvtColor(dst_warp,cv2.COLOR_BGR2GRAY)

    #################################################################
    ##########Ec(τ(q))=||tar(q)+▽Itar(q)τ(q)−Iref(q)||^2############
    ################################################################
    #todo processs the first condition
    #todo for every match sample point
    #first calculate the gradient of every point
    grad = calclulate_gradient(sample_vertices_or,src_warp_gray)
    #second calculate every b
    bs = []
    for i in range(sample_vertices_or.shape[0]):
        # intensity difference
        b1 = int(dst_warp_gray[sample_vertices_or[i,1],sample_vertices_or[i,0]])-int(src_warp_gray[sample_vertices_or[i,1],sample_vertices_or[i,0]])
        b2 = grad[i,0]*sample_vertices_or[i,0]+grad[i,1]*sample_vertices_or[i,1]
        b = b2-b1
        bs.append(b)
    bs = np.array(bs,dtype=np.float32)
    # thrid calculate every cofficient
    cofficients = []

    for j in range(sample_vertices_or.shape[0]):
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
    src_warp_gradient = calculate_gradient_graph(src_warp_gray)
    dst_warp_gradient = calculate_gradient_graph(dst_warp_gray)
    grad2 = gradient_gradient(sample_vertices_or,src_warp_gradient)
    ###########################################################################
    ############ E = ∑||Gt(pi+τ(pi))−Gs(pi)||^2################################
    ###########################################################################
    cofficients_g = []
    bbss = []
    for i in range(sample_vertices_or.shape[0]):
        bb = grad2[i]*sample_vertices_or[i] + src_warp_gradient[sample_vertices_or[i,1],sample_vertices_or[i,0]] - dst_warp_gradient[sample_vertices_or[i,1],sample_vertices_or[i,0]]
        bbss.append(bb)
        for w in weight[i]:
            cofficient_g = [w*grad2[i,0],w*grad2[i,1]]
            cofficients_g.append(cofficient_g)
    cofficients_g = np.array(cofficients_g)
    cofficients_g = cofficients_g.reshape(-1,4,2)
    bbss = np.array(bbss)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>")
    print(cofficients_g.shape)
    print(bbss.shape)


    #todo show all of these triangle
    # bg  = np.zeros_like(dst_warp)
    # for i in range(triangles.shape[0]):
    #     triangle = np.array([[mesh_boxes_dst[triangles[i, 0, 0], triangles[i, 0, 1]][0],
    #                           mesh_boxes_dst[triangles[i, 0, 0], triangles[i, 0, 1]][1]],
    #                          [mesh_boxes_dst[triangles[i, 1, 0], triangles[i, 1, 1]][0],
    #                           mesh_boxes_dst[triangles[i, 1, 0], triangles[i, 1, 1]][1]],
    #                          [mesh_boxes_dst[triangles[i, 2, 0], triangles[i, 2, 1]][0],
    #                           mesh_boxes_dst[triangles[i, 2, 0], triangles[i, 2, 1]][1]]],
    #                         dtype=np.int)
    #     cv2.fillConvexPoly(bg, triangle, (i*20 % 255, i*20 % 255, i*20 % 255))
    # cv2.imshow("bg", bg)

    # print("weight shape" + str(weight.shape))

    #
    # for i in range(location.shape[0]):
    #     ver1 = mesh_boxes_dst[location[i,0],location[i,1]]
    #     ver2 = mesh_boxes_dst[location[i,0],location[i,1]+1]
    #     ver3 = mesh_boxes_dst[location[i,0]+1,location[i,1]+1]
    #     ver4 = mesh_boxes_dst[location[i,0]+1,location[i,1]]
    #
    #     point = cofficients[i,0].dot(ver1) + cofficients[i,1].dot(ver2) + cofficients[i,2].dot(ver3) + cofficients[i,3].dot(ver4)
    #     origin_point = ver1*weight[i,0] + ver2*weight[i,1] + ver3*weight[i,2] + ver4*weight[i,3]
    #
    #     bbb = grad[i,:].dot(sample_vertices_or[i,:])
    #     aaa = grad[i,:].dot(origin_point)
    #     re = point - bs[i]
        # print(bbb-bs[i],re)
        # print(origin_point-sample_vertices_or[i])
    #
    # cofficients_g = cofficients_g*0
    # bbss = bbss*0
    print(">>>>>>>>>>>>>>>>>>>>>>>>>")
    print(triangles.shape)
    print(triangle_coefficient.shape)
    print(cofficients.shape)
    print(location.shape)
    print(bs.shape)
    print(mesh_boxes_src.shape)
    print(cofficients_g.shape)
    print(bbss.shape)
    # cofficients_g = cofficients_g*0
    # bbss = bbss*0
    # triangle_coefficient = triangle_coefficient*0
    c = optimization.optimize(triangles,triangle_coefficient,cofficients,location,bs,mesh_boxes_src,cofficients_g,bbss,0.16)
    c = c.astype(np.int)
    c = c.reshape(dst_y_num,dst_x_num,2)
    # c = c[:,:,(1,0)]
    # ccc = copy.deepcopy(c)
    # ccc = ccc.reshape(-1,2)
    # for i in range(ccc.shape[0]):
    #     ccc[i] = ccc[i]+[112,114]


    """the offset of texture_mapping bring"""
    offset_x = abs(min(np.min(c[:,:,0]),0))
    offset_y = abs(min(np.min(c[:,:,1]),0))


    # todo get the image after warping
    final_result = texture_mapping.texture_mapping(mesh_boxes_src.astype(np.int), c.astype(np.int),
                                    src_warp)
    final_result = final_result.astype(np.uint8)
    cv2.imshow("final_result",final_result)



    # todo show the point after warping
    warping_point = np.zeros_like(final_result)
    pointss = c.reshape(-1,2)
    for i in range(pointss.shape[0]):
        pointss[i] = pointss[i]+[offset_y,offset_x]
    pointss = pointss.astype(np.int)

    warping_point = draw.draw(warping_point,pointss)
    cv2.imshow("warp_point",warping_point)

    bg = np.zeros_like(final_result)

    # print(bg.shape)
    #     # print(src_warp.shape)
    #     # print(offset_x,offset_y)

    bg[offset_y:offset_y+src_warp.shape[0],offset_x:offset_x+src_warp.shape[1],:] = dst_warp
    # bg[offset_y:src_warp.shape[0]+offset_y,offset_x:src_warp.shape[1]+offset_x,:] = src_warp

    result2, mask = blending_average(final_result, bg)
    cv2.imshow("result2",result2)

    # point_distribute = np.zeros_like(final_result)
    # point_distribute_pic = draw.draw(src=final_result,src_point=ccc)
    # cv2.imshow("point_distribute_pic",point_distribute_pic)

    mesh_pic = draw.draw(src_warp,mesh_boxes_src.reshape(-1,2))

    cv2.imshow("mesh_pic",mesh_pic)
    cv2.imshow("src_warp",src_warp)
    cv2.imshow("dst_warp",dst_warp)
    cv2.imshow("result",result)
    cv2.waitKey(0)