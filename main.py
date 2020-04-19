from copy import deepcopy
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

import gcransac as gc


def draw_detected_feature(kps1, kps2, img1, img2, matches):
    plt.figure(figsize=(12, 8))

    img1 = cv2.drawKeypoints(img1, kps1, None, (0, 255, 0), 2)
    img2 = cv2.drawKeypoints(img2, kps2, None, (0, 255, 0), 2)
    img_out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)
    plt.imshow(img_out)

    plt.show()
    return


def draw_compare_matches(kps1, kps2, matches, img1, img2, cv_H, cv_mask, gc_H, gc_mask, show_trans=False):
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    if show_trans:
        # points transformation
        dst_gc = cv2.perspectiveTransform(pts, gc_H)
        dst_cv = cv2.perspectiveTransform(pts, cv_H)

        # blue is gc-ransac estimated, green is cv2-ransac estimated
        img2 = cv2.polylines(img2, [np.int32(dst_gc)], True, (0, 0, 255), 3, cv2.LINE_AA)
        img2 = cv2.polylines(img2, [np.int32(dst_cv)], True, (0, 255, 0), 3, cv2.LINE_AA)

    plt.figure(figsize=(12, 8))
    # draw match lines for cv2-ransac
    draw_params = dict(matchColor=(255, 255, 0),
                       singlePointColor=None,
                       matchesMask=cv_mask.ravel().tolist(),
                       flags=2)
    img_out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, **draw_params)
    ax1 = plt.subplot(2, 1, 1)
    plt.title("CV2-RANSAC")
    plt.imshow(img_out)

    # draw match lines for gc-ransac
    draw_params = dict(matchColor=(255, 255, 0),
                       singlePointColor=None,
                       matchesMask=gc_mask.ravel().tolist(),
                       flags=2)
    img_out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, **draw_params)
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
        src_path = 'img/fountain/fountain1.jpg'
        dst_path = 'img/fountain/fountain2.jpg'
    
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
    gc_H, gc_mask = gc.findHomography(src_pts, dst_pts, h1, w1, h2, w2, threshold=threshold, max_iters=5000)
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
    gc_H, gc_mask = gc.findFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, threshold=threshold, max_iters=5000)
    print('Inlier number = ', deepcopy(gc_mask).astype(np.float32).sum())
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


if __name__ == "__main__":
    dataset = "fountain"
    src_img, dst_img = load_test_datasets(dataset)

    src_K = np.loadtxt('img/fountain/fountain1.K')
    dst_K = np.loadtxt('img/fountain/fountain2.K')

    # 创建 ORB 特征提取器
    detetor = cv2.ORB_create(2000)
    # 提取 ORB 角点特征点 keypoints，特征点提取区域局部图像 descriptions
    keypoints1, descriptions1 = detetor.detectAndCompute(src_img, None)
    keypoints2, descriptions2 = detetor.detectAndCompute(dst_img, None)
    # BF 暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 特征描述子匹配
    matches = bf.match(descriptions1, descriptions2)
    # 特征描述子排序筛选 一半匹配
    matches = sorted(matches, key=lambda x: x.distance)[:len(matches) // 2]

    # 输出初始获取的暴力匹配结果
    print(f"Detect {dataset} features")
    print(f"Features found in src image = {len(keypoints1)}")
    print(f"Features found in dst image = {len(keypoints2)}")
    print(f"Matches number = {len(matches)}\n")
    draw_detected_feature(keypoints1, keypoints2, src_img, dst_img, matches)

    # 根据匹配结果构建点对
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    # 获取图像长宽信息
    h1, w1, _ = np.shape(src_img)
    h2, w2, _ = np.shape(dst_img)
    
    #cv_H, cv_mask, gc_H, gc_mask = testHomography(src_pts, dst_pts, h1, w1, h2, w2, 2.0)
    cv_H, cv_mask, gc_H, gc_mask = testFundamentalMat(src_pts, dst_pts, h1, w1, h2, w2, 1.0)
    #cv_H, cv_mask, gc_H, gc_mask = testEssentialMat(src_pts, dst_pts, src_K, dst_K, h1, w1, h2, w2, 1.0)

    # 绘制 cv-ransac gc-ransac 匹配结果对比图
    draw_compare_matches(keypoints1, keypoints2, matches, src_img, dst_img, cv_H, cv_mask, gc_H, gc_mask, True)
