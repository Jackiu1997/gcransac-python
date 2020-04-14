from copy import deepcopy
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

import gcransac as gc


def decolorize(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

def draw_matches(kps1, kps2, tentatives, img1, img2, H, mask, title):
    if H is None:
        print("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    # Ground truth transformation
    dst_GT = cv2.perspectiveTransform(pts, H_gt)
    img2_tr = cv2.polylines(decolorize(img2), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
    img2_tr = cv2.polylines(deepcopy(img2_tr), [np.int32(dst_GT)], True, (0, 255, 0), 3, cv2.LINE_AA)

    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img_out = cv2.drawMatches(decolorize(img1), kps1,
                              img2_tr, kps2, tentatives, None, **draw_params)
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.imshow(img_out)
    return

def homography_cv2(kps1, kps2, tentatives):
    src_pts = np.float32(
        [kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask

def homography_gcransac(kps1, kps2, tentatives, h1, w1, h2, w2):
    src_pts = np.float32(
        [kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32(
        [kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    H, mask = gc.findHomography(src_pts, dst_pts, h1, w1, h2, w2, 1.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask

if __name__ == "__main__":
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread('img/grafA.png'), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread('img/grafB.png'), cv2.COLOR_BGR2RGB)
    # 加载相机位移参数
    H_gt = np.linalg.inv(np.loadtxt('img/graf_model.txt'))

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

    testHomography()
