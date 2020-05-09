from copy import deepcopy
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygcransac as pygc
import gcransac as gc
import math as m
import matplotlib as mpl


dataset = 'zoom'
gt_M, vpts = None, None


def getReprojectionError(vpts, M):
    error = 0
    vpts_num = np.shape(vpts)[1]
    for i in range(vpts_num):
        src_x = vpts[0:3, i]
        dst_x = vpts[3:6, i]
        # 重投影并归一化
        re_x = np.dot(M, dst_x)
        error += np.sum(np.square(re_x / re_x[2] - src_x))
    return error / vpts_num


def getSampsonError(vpts, M):
    error = 0
    vpts_num = np.shape(vpts)[1]
    for i in range(vpts_num):
        src_x = vpts[0:3, i]
        dst_x = vpts[3:6, i]
        error += abs(np.dot(np.dot(src_x.T, M), dst_x))
    return error / vpts_num


def draw_detected_feature(kps1, kps2, img1, img2, matches):
    img1 = cv2.drawKeypoints(img1, kps1, None, (0, 255, 0), 2)
    img2 = cv2.drawKeypoints(img2, kps2, None, (0, 255, 0), 2)
    img_out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)
    return img_out


def draw_compare_matches(kps1, kps2, matches, img1, img2, M, mask, cmp_M):
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    # points transformation
    dst_cmp = cv2.perspectiveTransform(pts, cmp_M)
    dst_m = cv2.perspectiveTransform(pts, M)

    # blue is M estimated, green is ground truth estimated
    img2 = cv2.polylines(img2, [np.int32(dst_m)], True, (0, 0, 255), 3, cv2.LINE_AA)
    img2 = cv2.polylines(img2, [np.int32(dst_cmp)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # draw match lines for M
    draw_params = dict(matchesMask=mask.ravel().tolist(), flags=2)
    img_out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, **draw_params)

    return img_out


def testHomography(src_pts, dst_pts, h1, w1, h2, w2, threshold=1.0):
    match_img_list = []
    H, mask = None, None
    for i in range(3):
        t = time()
        if i == 0:
            print('CV2-RANSAC')
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, confidence=0.95, ransacReprojThreshold=threshold, maxIters=5000)
        elif i == 1:
            print('RHO-PROSAC')
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, confidence=0.95, ransacReprojThreshold=threshold, maxIters=5000)
        else:
            print('GC-RANSAC')
            H, mask = gc.findHomography(src_pts, dst_pts, h1, w1, h2, w2, threshold=threshold, max_iters=5000)
        print('Inlier number = ', deepcopy(mask).astype(np.float32).sum() / np.shape(src_pts)[0])
        print('Elapsed time = ', time()-t)
        print('Error = ', getReprojectionError(vpts, H), '\n')
        match_img = draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, H, mask, gt_M)
        match_img_list.append(match_img)

    # 绘制 cv-ransac gc-ransac 匹配结果对比图
    plt.figure(figsize=(12,8))
    mpl.rcParams.update({'font.size': 8})
    plt.subplot(3, 1, 1)
    plt.title("cv-ransac")
    plt.imshow(match_img_list[0])
    plt.subplot(3, 1, 2)
    plt.title("rho-prosac")
    plt.imshow(match_img_list[1])
    plt.subplot(3, 1, 3)
    plt.title("gc-ransac")
    plt.imshow(match_img_list[2])

    plt.savefig(f'results/H/{dataset}_H.png')
    plt.show()


def testFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, threshold=1.0):
    match_img_list = []
    F, mask = None, None
    for i in range(2):
        t = time()
        if i == 0:
            print('CV2-RANSAC')
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, confidence=0.95, ransacReprojThreshold=threshold)
        else:
            print('GC-RANSAC')
            #F, mask = gc.findFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, conf = 0.95, threshold=threshold)
            F, mask = pygc.findFundamentalMatrix(src_pts, dst_pts, h1, w1, h2, w2, conf = 0.95, threshold=threshold)
        print('Inlier number = ', deepcopy(mask).astype(np.float32).sum() / np.shape(src_pts)[0])
        print('Elapsed time = ', time()-t)
        print('Error = ', getSampsonError(vpts, F), '\n')
        match_img = draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, F, mask, gt_M)
        match_img_list.append(match_img)

    # 绘制 cv-ransac gc-ransac 匹配结果对比图
    plt.figure(figsize=(12,8))
    mpl.rcParams.update({'font.size': 8})
    plt.subplot(2, 1, 1)
    plt.title("cv-ransac")
    plt.imshow(match_img_list[0])
    plt.subplot(2, 1, 2)
    plt.title("gc-ransac")
    plt.imshow(match_img_list[1])

    plt.savefig(f'results/F/{dataset}_F.png')
    plt.show()


def testEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, threshold=1.0):
    match_img_list = []
    E, mask = None, None
    for i in range(3):
        t = time()
        if i == 0:
            print('CV2-RANSAC')
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, src_K, cv2.RANSAC, prob=0.95, threshold=threshold)
        elif i == 1:
            print('CV2-LMEDS')
            E, mask = cv2.findEssentialMat(src_pts, dst_pts, src_K, cv2.LMEDS)
        else:
            print('GC-RANSAC')
            E, mask = gc.findEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, threshold=threshold)
        print('Inlier number = ', deepcopy(mask).astype(np.float32).sum() / np.shape(src_pts)[0])
        print('Elapsed time = ', time()-t)
        print('Error = ', getSampsonError(vpts, E), '\n')
        match_img = draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, E, mask, gt_M)
        match_img_list.append(match_img)

    # 绘制 cv-ransac gc-ransac 匹配结果对比图
    plt.figure(figsize=(12,8))
    mpl.rcParams.update({'font.size': 8})
    plt.subplot(3, 1, 1)
    plt.title("cv-ransac")
    plt.imshow(match_img_list[0])
    plt.subplot(3, 1, 2)
    plt.title("rho-prosac")
    plt.imshow(match_img_list[1])
    plt.subplot(3, 1, 3)
    plt.title("gc-ransac")
    plt.imshow(match_img_list[2])

    plt.savefig(f'results/E/{dataset}_E.png')
    plt.show()


def load_evd_datasets(data_name):
    src_path = f'img/EVD/1/{data_name}.png'
    dst_path = f'img/EVD/2/{data_name}.png'
    gt_H = np.loadtxt(f'img/EVD/h/{data_name}.txt')
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img, gt_H


def load_homogr_datasets(data_name):
    src_path = f'img/homogr/{data_name}A.png'
    dst_path = f'img/homogr/{data_name}B.png'
    gt_H = np.loadtxt(f'img/homogr/{data_name}_H.txt')
    vpts = np.loadtxt(f'img/homogr/{data_name}_vpts.txt')
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img, gt_H, vpts


def load_kusvod2_datasets(data_name):
    src_path = f'img/kusvod2/{data_name}B.png'
    dst_path = f'img/kusvod2/{data_name}A.png'
    gt_M = np.loadtxt(f'img/kusvod2/{data_name}_M.txt')
    vpts = np.loadtxt(f'img/kusvod2/{data_name}_vpts.txt')
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img, gt_M, vpts


def load_strecha_datasets(data_name):
    src_path = f'img/strecha/{data_name}A.jpg'
    dst_path = f'img/strecha/{data_name}B.jpg'
    gt_M = np.zeros((3,3))
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img, gt_M


if __name__ == "__main__":
    src_img, dst_img, gt_M, vpts = load_kusvod2_datasets(dataset)

    # 创建 ORB 特征提取器
    detetor = cv2.xfeatures2d.SIFT_create(1000)
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
    
    # 测试单应矩阵
    #testHomography(src_pts, dst_pts, h1, w1, h2, w2, 1.0)

    # 测试基础矩阵
    testFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, 1.0)

    # 测试基础矩阵
    #testEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, 1.0)
    
