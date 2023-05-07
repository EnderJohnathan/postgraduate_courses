import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

FLANN_DISTANCE_THRESHOLD = 0.75
EXP = 3
RESIZE = True

current_dir = os.path.dirname(os.path.abspath(__file__))
anchor_image_path = os.path.join(os.path.dirname(current_dir) , "inputs\\inputs{}\\1.jpg".format(EXP))
sample_image_path = os.path.join(os.path.dirname(current_dir) , "inputs\\inputs{}\\2.jpg".format(EXP))

save_path = os.path.join(os.path.join(os.path.dirname(current_dir), "images\\result{}".format(EXP) ,'sift_flann_FLANNThres{}.png'.format(FLANN_DISTANCE_THRESHOLD)))

def read_img(image_path, txt_path=None, label=None):
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 使用imdecode函数进行读取
    with open(image_path, 'rb') as f:
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
    # 解码图像数据
    image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

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

def sift(img1, img2):
    t1 = time.time()
    sift = cv2.SIFT.create(2000) # sift
    # sift = cv2.ORB_create(2000) # orb
    # 计算特征点和描述子
    kps1, des1 = sift.detectAndCompute(img1, None)
    kps2, des2 = sift.detectAndCompute(img2, None)
    # 展示描述子
    img1Show = cv2.drawKeypoints(img1, kps1, img1)
    img2Show = cv2.drawKeypoints(img2, kps2, img2)

    plt.subplots_adjust(bottom=0.5)
    plt.subplot(1,3,1)
    plt.title("image 1")
    plt.imshow(img1Show)
    plt.subplot(1,3,2)
    plt.title("image 2")
    plt.imshow(img2Show)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < FLANN_DISTANCE_THRESHOLD * n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),

                    matchesMask = matchesMask,
                    flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kps1,img2,kps2,matches,None,**draw_params)
    t2 = time.time()
    plt.subplots_adjust(bottom=0.1, right=0.7, left = 0.1, top=1)
    cax = plt.axes([0.6, 0, 0.35, 1])
    plt.title("FLANN matching")
    plt.imshow(img3)
    plt.show()
    print("Time spent for detecting keypoints for one image is: {}s".format(t2-t1))
    
    print(save_path)
    cv2.imwrite(save_path, cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))


def draw_imshow(name, img, w=400, h=900):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name,img)

def main():
    # anchor_img = read_img(anchor_image_path, None, "0")
    # sample_img = read_img(sample_image_path, None, "0")
    anchor_img = read_img(anchor_image_path)
    sample_img = read_img(sample_image_path)
    if RESIZE:
        height, width = sample_img.shape
        anchor_img = cv2.resize(anchor_img, (width, height))

    h1, w1 = anchor_img.shape[:]
    h2, w2 = sample_img.shape[:]
    # cv2.namedWindow("anchor_img", 0)
    # cv2.resizeWindow("anchor_img", w1,h1) 
    
    # draw_imshow("anchor_img", anchor_img)

    # cv2.namedWindow("sample_img", 0)
    # cv2.resizeWindow("sample_img", w2,h2) 
    # cv2.imshow("sample_img", sample_img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    sift(anchor_img, sample_img)


if __name__ == "__main__":
    main()