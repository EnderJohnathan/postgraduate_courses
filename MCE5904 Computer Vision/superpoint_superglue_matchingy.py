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
from superglue import *


FLANN_DISTANCE_THRESHOLD = 0.7
SUPERGLUE_MATCHING_THRESHOLD = 0.2
RESIZE = True


current_dir = os.path.dirname(os.path.abspath(__file__))
anchor_image_path = os.path.join(os.path.dirname(current_dir) , "inputs\inputs1\wscottheath_512796821.rd.jpg")
sample_image_path = os.path.join(os.path.dirname(current_dir) , "inputs\inputs1\yisris_267944204.rd.jpg")

save_path = os.path.join(os.path.join(os.path.dirname(current_dir), "images\\result\\superpoint-superglue" ,'superpoint_superglue_FLANNThres{}.png'.format(FLANN_DISTANCE_THRESHOLD)))

def draw_imshow(name, img, w=800, h=600):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name,img)

def to_superglue_format(keypoints, descriptors, score, label):
    keypoints_np = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
    descriptors_np = np.array(descriptors)
    score_np = np.array(score)
    data = {
        'keypoints{}'.format(label): torch.tensor(keypoints_np, dtype=torch.float32).unsqueeze(0),
        'descriptors{}'.format(label): torch.tensor(descriptors_np, dtype=torch.float32).unsqueeze(0),
        'scores{}'.format(label): torch.tensor(score_np, dtype=torch.float32).unsqueeze(0)
    }
    return data

# def to_superglue_format(keypoints, descriptors, score, label):
#     keypoints_np = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
#     descriptors_np = np.array(descriptors)
#     score_np = np.array(score)
#     score_np = np.expand_dims(score_np, axis=0)
#     score_np = np.expand_dims(score_np, axis=0)
#     data = {
#         'keypoints{}'.format(label): torch.tensor(keypoints_np, dtype=torch.float32).unsqueeze(0),
#         'descriptors{}'.format(label): torch.tensor(descriptors_np, dtype=torch.float32).unsqueeze(0),
#         'scores{}'.format(label): torch.tensor(score_np, dtype=torch.float32)
#     }
#     return data

if __name__ == '__main__':
    anchor_img = cv2.imread(anchor_image_path, cv2.IMREAD_GRAYSCALE) # image 2
    sample_img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE) # image 1
    if RESIZE:
        height, width = sample_img.shape
        anchor_img = cv2.resize(anchor_img, (width, height))
        # sample_img = cv2.resize(sample_img, (1600, 1200))

    print('==> Loading pre-trained SuperPoint network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = superpoint.SuperPointFrontend(weights_path=os.path.join(current_dir, "weights/superpoint_v1.pth"),
                                       nms_dist=4,
                                       conf_thresh=0.005,
                                       nn_thresh=0.7,
                                       cuda=True)
    print('==> Successfully loaded pre-trained SuperPoint network.')

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

    keypoints_img1, descriptors_img1, h1 = fe.run(image1)
    keypoints_img2, descriptors_img2, h2 = fe.run(image2)

    # to transfer array ==> KeyPoints
    keypoints_img1 = [cv2.KeyPoint(keypoints_img1[0][i], keypoints_img1[1][i], 1)
                     for i in range(keypoints_img1.shape[1])]
    keypoints_img2 = [cv2.KeyPoint(keypoints_img2[0][i], keypoints_img2[1][i], 1)
                       for i in range(keypoints_img2.shape[1])]
    print("The number of keypoints in image1 is", len(keypoints_img1))
    print("The number of keypoints in image2 is", len(keypoints_img2))

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    flags = 0)

    # ************************************************
    # ***SuperGlue matchings***
    print('==> Loading pre-trained SuperGlue network.')
    superglue = SuperGlue(config={
        'descriptor_dim': 256,
        'weights': 'indoor',
        'sinkhorn_iterations': 50,
        'match_threshold': SUPERGLUE_MATCHING_THRESHOLD,
        'cuda': True
    })
    print('==> Successfully loaded pre-trained SuperGlue {} weight network.'.format(superglue.config['weights']))

    data1 = to_superglue_format(keypoints_img1, descriptors_img1, h1, 0)
    data2 = to_superglue_format(keypoints_img2, descriptors_img2, h2, 1)
    data = {**data1, **data2, 'image0': image1_origin, 'image1': image2_origin}
    print("*************")
    print("keypoints0 shape:{}, descriptor0 shape:{}, score0 shape:{}".format(data['keypoints0'].shape, data['descriptors0'].shape, data['scores0'].shape))
    print("keypoints1 shape:{}, descriptor1 shape:{}, score1 shape:{}".format(data['keypoints1'].shape, data['descriptors1'].shape, data['scores1'].shape))

    matches, confidences = superglue.forward(data)
    good_matches = [cv2.DMatch(i, j, 0) for i, j in enumerate(matches[0]) if j > -1]
    print("The number of good matches is", len(good_matches))
    img_matches_superglue = np.empty((max(image1_origin.shape[0], image2_origin.shape[0]), image1_origin.shape[1] + image2_origin.shape[1], 3),
                           dtype=np.uint8)
    cv2.drawMatches(image1_origin, keypoints_img1, image2_origin, keypoints_img2, good_matches, img_matches_superglue,
                    **draw_params)
    t2 = time.time()

    # ************************************************

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
    axs[0, 2].imshow(img_matches_superglue)
    axs[0, 2].set_title('Good SuperGlue Matches of SuperPoint')

    # Save results
    cv2.imwrite(os.path.join(os.path.join(os.path.dirname(current_dir), "images\\result\\kps" ,'superpoint_image1_kps.png')), sample_img)
    cv2.imwrite(os.path.join(os.path.join(os.path.dirname(current_dir), "images\\result\\kps" ,'superpoint_image2_kps.png')), anchor_img)
    cv2.imwrite(save_path, cv2.cvtColor(img_matches_superglue, cv2.COLOR_BGR2RGB))

    axs[1, 2].axis('off')  # Remove the last subplot

    # Show the plot
    plt.show()

    cv2.waitKey(0)
    print("Time spent for detecting keypoints for one image is: {}s".format(t2-t1))
