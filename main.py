from copy import deepcopy
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygcransac as pygc
import gcransac as gc
import math as m

def getHomographyError(src_img, dst_img, gt_M, cv_M, gc_M):
    h, w, ch = src_img.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # points transformation
    dst_gt = cv2.perspectiveTransform(pts, gt_M)
    dst_cv = cv2.perspectiveTransform(pts, cv_M)
    dst_gc = cv2.perspectiveTransform(pts, gc_M)
    e1 = (dst_gt[0][0,0] - dst_cv[0][0,0])**2 + (dst_gt[0][0,1] - dst_cv[0][0,1])**2
    e2 = (dst_gt[0][0,0] - dst_gc[0][0,0])**2 + (dst_gt[0][0,1] - dst_gc[0][0,1])**2
    print(m.sqrt(e1))
    print(m.sqrt(e2))


def getSampsonError(src_img, dst_img, gt_M, cv_M, gc_M):
    vpts = np.loadtxt('img/kusvod2/booksh_vpts.txt')
    src_x = vpts[0:3, 0]
    dst_x = vpts[3:6, 0]
    gt_e = np.dot(np.dot(src_x.T, gt_M), dst_x)
    cv_e = np.dot(np.dot(src_x.T, cv_M), dst_x)
    gc_e = np.dot(np.dot(src_x.T, gc_M), dst_x)
    print(abs(cv_e))
    print(abs(gc_e))


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
    # 测试 cv-ransac
    print('CV2-RANSAC')
    t = time()
    cv_H, cv_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=5000)
    print('Inlier number = ', deepcopy(cv_mask).astype(np.float32).sum() / np.shape(src_pts)[0])
    print('Elapsed time = ', time()-t, '\n')

    # 测试 gc-ransac
    print('GC-RANSAC')
    t = time()
    gc_H, gc_mask = gc.findHomography(src_pts, dst_pts, h1, w1, h2, w2, threshold=threshold, max_iters=5000)
    print('Inlier number = ', deepcopy(gc_mask).astype(np.float32).sum() / np.shape(src_pts)[0])
    print('Elapsed time = ', time()-t, '\n')

    return cv_H, cv_mask, gc_H, gc_mask


def testFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, threshold=1.0):
    # 测试 cv-ransac
    print('CV2-RANSAC')
    t = time()
    cv_H, cv_mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=threshold)
    print('Inlier number = ', deepcopy(cv_mask).astype(np.float32).sum() / np.shape(src_pts)[0])
    print('Elapsed time = ', time()-t, '\n')

    # 测试 gc-ransac
    print('GC-RANSAC')
    t = time()
    gc_H, gc_mask = gc.findFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, threshold=threshold, max_iters=1000)
    print('Inlier number = ', deepcopy(gc_mask).astype(np.float32).sum() / np.shape(src_pts)[0]) 
    print('Elapsed time = ', time()-t, '\n')

    return cv_H, cv_mask, gc_H, gc_mask


def testEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, threshold=1.0):
    # 测试 cv-ransac
    print('CV2-RANSAC')
    t = time()
    cv_H, cv_mask = cv2.findEssentialMat(src_pts, dst_pts, src_K, cv2.RANSAC, prob=0.99, threshold=threshold)
    print('Inlier number = ', deepcopy(cv_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t, '\n')

    # 测试 gc-ransac
    print('GC-RANSAC')
    t = time()
    gc_H, gc_mask = gc.findEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, threshold=threshold, max_iters=5000)
    print('Inlier number = ', deepcopy(gc_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t)

    return cv_H, cv_mask, gc_H, gc_mask


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
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img, gt_H


def load_kusvod2_datasets(data_name):
    src_path = f'img/kusvod2/{data_name}B.png'
    dst_path = f'img/kusvod2/{data_name}A.png'
    gt_M = np.loadtxt(f'img/kusvod2/{data_name}_M.txt')
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img, gt_M


def load_strecha_datasets(data_name):
    src_path = f'img/strecha/{data_name}A.jpg'
    dst_path = f'img/strecha/{data_name}B.jpg'
    gt_M = np.zeros((3,3))
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img, gt_M


if __name__ == "__main__":
    dataset = "box"
    src_img, dst_img, gt_M = load_kusvod2_datasets(dataset)

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

    '''
    src_pts = np.vstack( (src_pts.T, np.ones((1, np.shape(src_pts)[0]))) )
    dst_pts = np.vstack( (dst_pts.T, np.ones((1, np.shape(dst_pts)[0]))) )
    points = np.vstack((src_pts, dst_pts))
    np.savetxt("points.txt", points)
    '''

    # 输出初始获取的暴力匹配结果
    print(f"Detect {dataset} features")
    print(f"Features found in src image = {len(keypoints1)}")
    print(f"Features found in dst image = {len(keypoints2)}")
    print(f"Matches number = {len(matches)}\n")
    '''
    plt.figure(figsize=(12,8))
    init_img = draw_detected_feature(keypoints1, keypoints2, src_img, dst_img, matches)
    plt.imshow(init_img)
    plt.show()
    '''
    
    ''' 测试单应矩阵
    cv_M, cv_mask, gc_M, gc_mask = testHomography(src_pts, dst_pts, h1, w1, h2, w2, 1.0)
    getHomographyError(src_img, dst_img, gt_M, cv_M, gc_M)
    '''

    ''' 测试基础矩阵 '''
    cv_M, cv_mask, gc_M, gc_mask = testFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, 1.0)
    getSampsonError(src_img, dst_img, gt_M, cv_M, gc_M)
    
    ''' 测试基础矩阵
    cv_M, cv_mask, gc_M, gc_mask = testEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, 1.0)
    getSampsonError(src_img, dst_img, gt_M, cv_M, gc_M)
    '''

    # 绘制 cv-ransac gc-ransac 匹配结果对比图
    plt.figure(figsize=(12,8))
    cv_img = draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, cv_M, cv_mask, gt_M)
    gc_img = draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, gc_M, gc_mask, gt_M)
    plt.subplot(2, 1, 1)
    plt.imshow(cv_img)
    plt.subplot(2, 1, 2)
    plt.imshow(gc_img)
    plt.show()
