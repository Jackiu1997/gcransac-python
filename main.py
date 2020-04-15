from copy import deepcopy
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

import gcransac as gc


def draw_matches(kps1, kps2, tentatives, img1, img2, H, mask, title):
    if H is None:
        print("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor=(255, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    img_out = cv2.drawMatches(img1, kps1, img2, kps2, tentatives, None, **draw_params)
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.imshow(img_out)
    return

def homography_cv2(kps1, kps2, tentatives):
    src_pts = np.float32(
        [kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask

def homography_gcransac(kps1, kps2, tentatives, h1, w1, h2, w2):
    src_pts = np.float32(
        [kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32(
        [kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    H, mask = gc.findHomography(src_pts, dst_pts, h1, w1, h2, w2, 2.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask

def fundamental_cv2(kps1, kps2, tentatives):
    src_pts = np.float32(
        [kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    H, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 3.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask

def fundamental_gcransac(kps1, kps2, tentatives, h1, w1, h2, w2):
    src_pts = np.float32(
        [kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32(
        [kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    H, mask = gc.findFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, 3.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask

def essential_gcransac(kps1, kps2, tentatives, src_K, dst_K, h1, w1, h2, w2):
    src_pts = np.float32(
        [kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32(
        [kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    H, mask = gc.findEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, 3.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask


if __name__ == "__main__":
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread('img/fountain/fountain1.jpg'), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread('img/fountain/fountain2.jpg'), cv2.COLOR_BGR2RGB)
    # 加载相机内参
    src_K = np.loadtxt('img/fountain/fountain1.K')
    dst_K = np.loadtxt('img/fountain/fountain2.K')

    # 创建 ORB 特征提取器
    detetor = cv2.ORB_create(1000)
    # 提取 ORB 角点特征点 keypoints，特征点提取区域局部图像 descriptions
    keypoints1, descriptions1 = detetor.detectAndCompute(src_img, None)
    keypoints2, descriptions2 = detetor.detectAndCompute(dst_img, None)
    # BF 暴力匹配结果
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # BF 匹配结果筛选结果
    tentatives = bf.match(descriptions1, descriptions2)

    def testHomography():
        # 测试 cv-ransac
        t = time()
        cv2_H, cv2_mask = homography_cv2(keypoints1,
                                         keypoints2,
                                         tentatives)
        print(time()-t, ' sec cv2')

        # 测试 gc-ransac
        t = time()
        mag_H, mag_mask = homography_gcransac(keypoints1,
                                              keypoints2,
                                              tentatives,
                                              src_img.shape[0],
                                              src_img.shape[1],
                                              dst_img.shape[0],
                                              dst_img.shape[1])
        print(time()-t, ' sec gc-ransac')

        # 绘制匹配结果图像
        draw_matches(keypoints1, keypoints2, tentatives, src_img, dst_img, cv2_H, cv2_mask, "CV2-RANSAC")
        draw_matches(keypoints1, keypoints2, tentatives, src_img, dst_img, mag_H, mag_mask, "GC-RANSAC")
        plt.show()

    def testFundamentalMat():
        # 测试 cv-ransac
        t = time()
        cv2_H, cv2_mask = fundamental_cv2(keypoints1,
                                          keypoints2,
                                          tentatives)
        print(time()-t, ' sec cv2')

        # 测试 gc-ransac
        t = time()
        mag_H, mag_mask = fundamental_gcransac(keypoints1,
                                               keypoints2,
                                               tentatives,
                                               src_img.shape[0],
                                               src_img.shape[1],
                                               dst_img.shape[0],
                                               dst_img.shape[1])
        print(time()-t, ' sec gc-ransac')

        # 绘制匹配结果图像
        draw_matches(keypoints1, keypoints2, tentatives, src_img, dst_img, cv2_H, cv2_mask, "CV2-RANSAC")
        draw_matches(keypoints1, keypoints2, tentatives, src_img, dst_img, mag_H, mag_mask, "GC-RANSAC")
        plt.show()
    
    def testEssentialMat():
        # 测试 gc-ransac
        t = time()
        mag_H, mag_mask = essential_gcransac(keypoints1,
                                             keypoints2,
                                             tentatives,
                                             src_K, 
                                             dst_K,
                                             src_img.shape[0],
                                             src_img.shape[1],
                                             dst_img.shape[0],
                                             dst_img.shape[1])
        print(time()-t, ' sec gc-ransac')

        # 绘制匹配结果图像
        draw_matches(keypoints1, keypoints2, tentatives, src_img, dst_img, mag_H, mag_mask, "GC-RANSAC")
        plt.show()
    
    while True:
        print("1.Homography\n2.FundamentalMatrix\n3.EssentialMatrix\n0:exit\nPlease input try to solve:")
        option = int(input())
        print("\n")
        if option == 1:
            testHomography()
        elif option == 2:
            testFundamentalMat()
        elif option == 3:
            testEssentialMat()
        elif option == 0:
            break
        else:
            continue
        print("\n")