# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   superpoint_detect_sixsubplots.py
@Time    :   2023/05/06 16:26:07
@Author  :   Liao Hongjie
@Version :   1.0
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import superpoint
import time
# from detect import *


FLANN_DISTANCE_THRESHOLD = 0.85
RESIZE = True
EXP = 4

current_dir = os.path.dirname(os.path.abspath(__file__))
anchor_image_path = os.path.join(os.path.dirname(current_dir) , "inputs\\inputs{}\\1.jpg".format(EXP))
sample_image_path = os.path.join(os.path.dirname(current_dir) , "inputs\\inputs{}\\2.jpg".format(EXP))


save_path = os.path.join(os.path.join(os.path.dirname(current_dir), "images\\result{}".format(EXP),'superpoint_flann_FLANNThres{}.png'.format(FLANN_DISTANCE_THRESHOLD)))

def draw_imshow(name, img, w=800, h=600):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name,img)

def read_img(image_path, txt_path=None, label=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    if txt_path != None:
        info_list = []

        txt = open(txt_path, "r")
        for line in txt.readlines():
            line = line.strip('\n')
            elements = line.split(" ")
            info_list.append(
                [elements[0], elements[1], elements[2], elements[3], elements[4]])
        txt.close()

        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        distance = 999
        for info in info_list:
            if info[0] == label:
                c_x = float(info[1])
                c_y = float(info[2])
                c_w = float(info[3])
                c_h = float(info[4])
                dis = abs(c_x - 0.5) + abs(c_y - 0.5)
                if dis < distance:
                    distance = dis
                    x1 = int((c_x - c_w / 2) * width)
                    y1 = int((c_y - c_h / 2) * height)
                    x2 = int((c_x + c_w / 2) * width)
                    y2 = int((c_y + c_h / 2) * height)
                    cut_image = image[y1:y2, x1:x2]
                    # cv2.imshow("cut image", cut_image)
        return cut_image
    else:
        return image


if __name__ == '__main__':
    # anchor_img = read_img(anchor_image_path, anchor_txt, "0")
    # sample_img = read_img(sample_image_path, sample_txt, "0")
    anchor_img = read_img(sample_image_path) # image 2
    sample_img = read_img(anchor_image_path) # image 1
    if RESIZE:
        height, width = sample_img.shape
        anchor_img = cv2.resize(anchor_img, (width, height))
        # sample_img = cv2.resize(sample_img, (1600, 1200))

    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = superpoint.SuperPointFrontend(weights_path=os.path.join(current_dir, "./weights/superpoint_v1.pth"),
                                       nms_dist=4,
                                       conf_thresh=0.005,
                                       nn_thresh=0.7,
                                       cuda=True)
    print('==> Successfully loaded pre-trained network.')

    image1_origin = sample_img
    image2_origin = anchor_img
    image1 = sample_img.astype(np.float32)
    image2 = anchor_img.astype(np.float32)
    image1 = image1 / 255.
    image2 = image2 / 255.

    if image1 is None or image2 is None:
        print('Could not open or find the images!')
        exit(0)

    t1 = time.time()
    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors

    keypoints_obj, descriptors_obj, h1 = fe.run(image1)
    keypoints_scene, descriptors_scene, h2 = fe.run(image2)

    # to transfer array ==> KeyPoints
    keypoints_obj = [cv2.KeyPoint(keypoints_obj[0][i], keypoints_obj[1][i], 1)
                     for i in range(keypoints_obj.shape[1])]
    keypoints_scene = [cv2.KeyPoint(keypoints_scene[0][i], keypoints_scene[1][i], 1)
                       for i in range(keypoints_scene.shape[1])]
    print("The number of keypoints in image1 is", len(keypoints_obj))
    print("The number of keypoints in image2 is", len(keypoints_scene))

    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj.T, descriptors_scene.T, 2)

    # -- Filter matches using the Lowe's ratio test
    
    good_matches = []
    for m, n in knn_matches:
        if m.distance < FLANN_DISTANCE_THRESHOLD * n.distance:
            good_matches.append(m)

    t2 = time.time()

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints_obj[match.queryIdx].pt
        points2[i, :] = keypoints_scene[match.trainIdx].pt

    # # Use homography
    # M, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # height, width = image2_origin.shape
    # im1Reg = cv2.warpPerspective(image1_origin, M, (width, height))
    # im1Reg_ori = im1Reg


    # 画匹配了的关键点
    X = points2  # 输入：待对齐图片的关键点坐标
    y = points1 # 输出：anchor的关键点坐标
    for i in range(X.shape[0]):
        point = X[i, :]
        cv2.circle(anchor_img, (int(point[0]), int(
            point[1])), 1, (255, 0, 0), 2)
    for i in range(y.shape[0]):
        point = y[i, :]
        cv2.circle(sample_img, (int(point[0]), int(
            point[1])), 1, (255, 255, 0), 2)

    # -- Draw matches
    img_matches = np.empty((max(image1_origin.shape[0], image2_origin.shape[0]), image1_origin.shape[1] + image2_origin.shape[1], 3),
                           dtype=np.uint8)
    # cv2.drawMatches(image1_origin, keypoints_obj, image2_origin, keypoints_scene, good_matches, img_matches,
    #                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # 不带关键点的匹配图
    
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    flags = 0)
    cv2.drawMatches(image1_origin, keypoints_obj, image2_origin, keypoints_scene, good_matches, img_matches,
                **draw_params) # 带关键点的匹配图

    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
    image1_origin = cv2.cvtColor(image1_origin, cv2.COLOR_BGR2RGB)
    image2_origin = cv2.cvtColor(image2_origin, cv2.COLOR_BGR2RGB)
    # img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    # Create a subplot with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Show the images in the subplots


    axs[0, 0].imshow(image1_origin)
    axs[0, 0].set_title('image 1')
    axs[0, 1].imshow(image2_origin)
    axs[0, 1].set_title('image 1')
    axs[1, 0].imshow(sample_img)
    axs[1, 0].set_title('image 1 with Matched Keypoints')
    axs[1, 1].imshow(anchor_img)
    axs[1, 1].set_title('image 2 with Matched Keypoints')
    axs[0, 2].imshow(img_matches)
    axs[0, 2].set_title('Good Matches of SuperPoint')

    # Save results
    cv2.imwrite(os.path.join(os.path.join(os.path.dirname(current_dir), "images\\result\\kps" ,'superpoint_image1_kps.png')), sample_img)
    cv2.imwrite(os.path.join(os.path.join(os.path.dirname(current_dir), "images\\result\\kps" ,'superpoint_image2_kps.png')), anchor_img)
    cv2.imwrite(save_path, cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))

    axs[1, 2].axis('off')  # Remove the last subplot

    # Show the plot
    plt.show()

    cv2.waitKey(0)
    print("Time spent for detecting keypoints for one image is: {}s".format(t2-t1))
