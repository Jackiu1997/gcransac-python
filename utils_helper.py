import numpy as np
import cv2


""" 误差计算模块（重投影、Sampson） """
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


""" 对比信息绘制模块 """
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


""" 数据集合加载模块 """
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