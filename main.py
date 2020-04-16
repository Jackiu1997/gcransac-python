from copy import deepcopy
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

import gcransac as gc


def draw_compare_matches(kps1, kps2, tentatives, img1, img2, cv_H, cv_mask, gc_H, gc_mask):
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    # points transformation
    dst_gc = cv2.perspectiveTransform(pts, gc_H)
    dst_cv2 = cv2.perspectiveTransform(pts, cv_H)

    # blue is gc-ransac estimated, green is cv2-ransac estimated
    img2_tr = cv2.polylines(img2, [np.int32(dst_gc)], True, (0, 0, 255), 3, cv2.LINE_AA)
    img2_tr = cv2.polylines(deepcopy(img2_tr), [np.int32(dst_cv2)], True, (0, 255, 0), 3, cv2.LINE_AA)

    plt.figure(figsize=(12, 8))
    # draw match lines for cv2-ransac
    draw_params = dict(matchColor=(255, 255, 0),
                       singlePointColor=None,
                       matchesMask=cv_mask.ravel().tolist(),
                       flags=2)
    img_out = cv2.drawMatches(img1, kps1, img2_tr, kps2, tentatives, None, **draw_params)
    ax1 = plt.subplot(2, 1, 1)
    plt.title("CV2-RANSAC")
    plt.imshow(img_out)

    # draw match lines for gc-ransac
    draw_params = dict(matchColor=(255, 255, 0),
                       singlePointColor=None,
                       matchesMask=gc_mask.ravel().tolist(),
                       flags=2)
    img_out = cv2.drawMatches(img1, kps1, img2_tr, kps2, tentatives, None, **draw_params)
    ax2 = plt.subplot(2, 1, 2)
    plt.title("GC-RANSAC")
    plt.imshow(img_out)

    plt.show()
    return


def load_test_datasets(data_name):
    if data_name == "adam":
        src_path = 'img/adam/adam1.png'
        dst_path = 'img/adam/adam2.png'
    elif data_name == "Eiffel":
        src_path = 'img/Eiffel/Eiffel1.png'
        dst_path = 'img/Eiffel/Eiffel2.png'
    elif data_name == "graf":
        src_path = 'img/graf/graf1.png'
        dst_path = 'img/graf/graf2.png'
    elif data_name == "head":
        src_path = 'img/head/head1.jpg'
        dst_path = 'img/head/head2.jpg'
    elif data_name == "johnssona":
        src_path = 'img/johnssona/johnssona1.jpg'
        dst_path = 'img/johnssona/johnssona2.jpg'
    elif data_name == "Kyoto":
        src_path = 'img/Kyoto/Kyoto1.jpg'
        dst_path = 'img/Kyoto/Kyoto2.jpg'
    elif data_name == "fountain":
        src_path = 'img/fountain/fountain1.png'
        dst_path = 'img/fountain/fountain2.png'
        src_K = np.loadtxt('img/fountain/fountain1.K')
        src_K = np.loadtxt('img/fountain/fountain2.K')
    
    # 加载 src_img dst_img
    src_img = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    dst_img = cv2.cvtColor(cv2.imread(dst_path), cv2.COLOR_BGR2RGB)

    return src_img, dst_img


def testHomography(src_pts, dst_pts, h1, w1, h2, w2, threshold=1.0):
    # 测试 cv-ransac
    print('CV2-RANSAC')
    t = time()
    cv_H, cv_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    print('Inlier number = ', deepcopy(cv_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t, '\n')

    # 测试 gc-ransac
    print('GC-RANSAC')
    t = time()
    gc_H, gc_mask = gc.findHomography(src_pts, dst_pts, h1, w1, h2, w2, threshold)
    print('Inlier number = ', deepcopy(gc_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t, '\n')

    return cv_H, cv_mask, gc_H, gc_mask


def testFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, threshold=1.0):
    # 测试 cv-ransac
    print('CV2-RANSAC')
    t = time()
    cv_H, cv_mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, threshold)
    print('Inlier number = ', deepcopy(cv_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t, '\n')

    # 测试 gc-ransac
    print('GC-RANSAC')
    t = time()
    gc_H, gc_mask = gc.findFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, threshold)
    print('Inlier number = ', deepcopy(cv_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t, '\n')

    return cv_H, cv_mask, gc_H, gc_mask


def testEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, threshold=1.0):
    # 测试 cv-ransac
    print('CV2-RANSAC')
    t = time()
    cv_H, cv_mask = cv2.findEssentialMat(src_pts, dst_pts, src_K, cv2.RANSAC, threshold)
    print('Inlier number = ', deepcopy(cv_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t, '\n')

    # 测试 gc-ransac
    print('GC-RANSAC')
    t = time()
    gc_H, gc_mask = gc.findEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, threshold)
    print('Inlier number = ', deepcopy(gc_mask).astype(np.float32).sum())
    print('Elapsed time = ', time()-t)

    return cv_H, cv_mask, gc_H, gc_mask


if __name__ == "__main__":
    src_img, dst_img = load_test_datasets('graf')

    # 创建 ORB 特征提取器
    detetor = cv2.ORB_create(1000)
    # 提取 ORB 角点特征点 keypoints，特征点提取区域局部图像 descriptions
    keypoints1, descriptions1 = detetor.detectAndCompute(src_img, None)
    keypoints2, descriptions2 = detetor.detectAndCompute(dst_img, None)
    # BF 暴力匹配结果
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # BF 匹配结果筛选结果
    tentatives = bf.match(descriptions1, descriptions2)

    # 根据匹配结果构建点对
    src_pts = np.float32(
        [keypoints1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32(
        [keypoints2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    # 获取图像长宽信息
    h1, w1, _ = np.shape(src_img)
    h2, w2, _ = np.shape(dst_img)
    
    cv_H, cv_mask, gc_H, gc_mask = testHomography(src_pts, dst_pts, h1, w1, h2, w2, 1.0)
    #cv_H, cv_mask, gc_H, gc_mask = testFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, 3.0)
    #cv_H, cv_mask, gc_H, gc_mask = testEssentialMat(src_pts, dst_pts, h1, w1, h2, w2, 3.0)
    draw_compare_matches(keypoints1, keypoints2, tentatives, src_img, dst_img, cv_H, cv_mask, gc_H, gc_mask)
