import math

import cv2
import numpy as np

from estimator import (EstimatorEssential, EstimatorFundamental,
                       EstimatorHomography)
from model import *
from sampler import UniformSampler
from solver import (SolverEssentialMatrixEightPoint,
                    SolverEssentialMatrixFivePointStewenius,
                    SolverFundamentalMatrixEightPoint,
                    SolverFundamentalMatrixSevenPoint,
                    SolverHomographyFourPoint)

from .ransac import RANSAC

def __transformInliersToMask(inliers, point_number):
    """ 转换 inliers 内点序号列表为 cv2 match 所需的 mask

    参数
    --------
    inliers : list
        内点序号列表
    point_number : int
        点集的数目
    
    返回
    --------
    list
        包含 0 1 的 mask 列表
    """
    mask = []
    for i in range(point_number):
        if i in inliers:
            mask.append(1)
        else:
            mask.append(0)
    mask = np.array(mask)
    return mask


""" 用于特征点匹配，对应矩阵求解的函数 """
def findHomography(src_points, dst_points, threshold=1.0, conf=0.95, max_iters=10000):
    """ 单应矩阵求解
    
    参数
    --------
    src_points : numpy
        源图像特征点集合
    dst_points : numpy
        目标图像特征点集合
    h1, w1: int, int
        源图像高度和宽度
    h2, w2: int, int
        目标图像高度和宽度
    threshold : float
        决定内点和外点的阈值
    conf : float
        RANSAC置信参数
    max_iters : int
        RANSAC算法最大迭代次数

    返回
    --------
    numpy, list
        基础矩阵，标注内点和外点的mask
    """
    # 合并points到同个矩阵：
    # src在前两列，dst在后两列
    points = np.c_[src_points, dst_points]

    ''' GC-RANSAC过程 '''
    # 设置模型估计器和模型
    estimator = EstimatorHomography(SolverHomographyFourPoint, 
                                    SolverHomographyFourPoint)
    model = Homography()

    # 设置全局采样
    main_sampler = UniformSampler(points)
    # 检查样本是否成功初始化
    if not main_sampler.initialized:
        print("采样器初始化失败\n")
        return None

    # 设置GC-RANSAC算法参数
    gcransac = RANSAC()
    gcransac.settings.threshold = threshold
    gcransac.settings.confidence = conf
    gcransac.settings.max_iteration_number = max_iters

    # 运行GC-RANSAC算法
    model, inliers = gcransac.run(points,
                                  estimator,
                                  main_sampler)

    print(f'Number of iterations = {gcransac.statistics.iteration_number}')

    # 获取特征点匹配结果（变换矩阵 和 模型对应内点）
    H = model.descriptor
    mask = __transformInliersToMask(inliers, gcransac.point_number)
    return H, mask
