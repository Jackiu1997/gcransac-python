from copy import deepcopy

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import gcransac as gc
from utils_helper import *

''' homogr
'adam', 'boat', 'Boston', 'BostonLib', 'BruggeSquare', 'BruggeTower', 'Brussels', 'CapitalRegion', 
'city', 'Eiffel', 'ExtremeZoom', 'graf', 'LePoint1', 'LePoint2', 'LePoint3', 'WhiteBoard'
'''
if __name__ == "__main__":
    dataset = 'adam'
    src_img, dst_img, gt_M, vpts = load_homogr_datasets(dataset)

    # 创建 ORB 特征提取器
    detetor = cv2.ORB_create(2000)
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
    print(f"Matches number = {len(matches)}", '\n')

    # 进行图像单应矩阵匹配估计
    H, mask = gc.findHomography(src_pts, dst_pts, h1, w1, h2, w2, threshold=1.0, max_iters=5000)
    print('Inliers Ratio = ', deepcopy(mask).astype(np.float32).sum() / np.shape(src_pts)[0])
    print('Error = ', getReprojectionError(vpts, H), '\n')

    # 绘制算法匹配结果图
    plt.figure(figsize=(12,8))
    mpl.rcParams.update({'font.size': 8})
    plt.title("gc-ransac")
    plt.imshow(draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, H, mask, gt_M))
    plt.show()
