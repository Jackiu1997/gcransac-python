import math as m
from copy import deepcopy
from time import time

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import gcransac as gc
from utils_helper import *

''' kusvod2
'booksh', 'box', 'castle', 'corr', 'graff', 'head', 'kampa', 'Kyoto', 
'leafs', 'plant', 'rotunda', 'shout', 'valbonne', 'wall', 'wash', 'zoom'
'''
if __name__ == "__main__":
    dataset = 'head'
    src_img, dst_img, gt_M, vpts = load_kusvod2_datasets(dataset)

    # 创建 ORB 特征提取器
    detetor = cv2.xfeatures2d.SIFT_create(2000)
    # 提取 ORB 角点特征点 keypoints，特征点提取区域局部图像 descriptions
    keypoints1, descriptions1 = detetor.detectAndCompute(src_img, None)
    keypoints2, descriptions2 = detetor.detectAndCompute(dst_img, None)

    # BF 暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptions1, descriptions2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 根据匹配结果构建点对
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    # 获取图像长宽信息 
    h1, w1, _ = np.shape(src_img)
    h2, w2, _ = np.shape(dst_img)
    
    # 输出初始获取的暴力匹配结果
    print(f"Detect {dataset} features")
    print(f"Features found in src image = {len(keypoints1)}")
    print(f"Features found in dst image = {len(keypoints2)}")
    print(f"Matches number = {len(matches)}", '\n')

    threshold = 1.0
    match_img_list = []
    F, mask = None, None
    for i in range(3):
        if i == 0:
            print('FM-RANSAC')
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, confidence=0.95, ransacReprojThreshold=threshold)
        else:
            print('GC-RANSAC')
            F, mask = gc.findFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, conf = 0.95, threshold=threshold, max_iters=5000)
        print('Inliers Ratio = ', deepcopy(mask).astype(np.float32).sum() / np.shape(src_pts)[0])
        print('Error = ', getSampsonError(vpts, F), '\n')
        match_img = draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, F, mask, gt_M)
        match_img_list.append(match_img)

    # 绘制 cv-ransac gc-ransac 匹配结果对比图
    plt.figure(figsize=(12,8))
    mpl.rcParams.update({'font.size': 8})
    plt.subplot(2, 1, 1)
    plt.title("fm-ransac")
    plt.imshow(match_img_list[0])
    plt.subplot(2, 1, 2)
    plt.title("gc-ransac")
    plt.imshow(match_img_list[1])

    plt.savefig(f'results/F/{dataset}_F.png')
    plt.show()