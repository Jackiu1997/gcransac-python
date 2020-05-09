import math

import cv2
import numpy as np

from estimator import (EstimatorEssential, EstimatorFundamental,
                       EstimatorHomography)
from model import *
from neighbor import GridNeighborhoodGraph
from sampler import ProgressiveNapsacSampler, UniformSampler, ProsacSampler
from solver import (SolverEssentialMatrixEightPoint,
                    SolverEssentialMatrixFivePointStewenius,
                    SolverFundamentalMatrixEightPoint,
                    SolverFundamentalMatrixSevenPoint,
                    SolverHomographyFourPoint)

from .gcransac import GCRANSAC


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


def __normalizeCorrespondences(points, intrinsics_src, intrinsics_dst):
    """ 通过内参矩阵归一化点集

    参数
    --------
    points : numpy
        数据点集
    intrinsics_src : numpy
        源图像内参矩阵
    intrinsics_dst : numpy
        目标图像内参矩阵

    返回
    --------
    numpy
        转换后的点集
    """
    normalized_points = points
    inverse_intrinsics_src = np.linalg.inv(intrinsics_src)
    inverse_intrinsics_dst = np.linalg.inv(intrinsics_dst)

    for i in range(np.shape(points)[0]):
        point = points[i]
        # Homogeneous point
        point_src = np.array([point[0], point[1], 1.0])
        point_dst = np.array([point[2], point[3], 1.0])

        # Normalized homogeneous point
        normalized_point_src = np.dot(inverse_intrinsics_src, point_src)
        normalized_point_dst = np.dot(inverse_intrinsics_dst, point_dst)

        normalized_points[i] = np.r_[normalized_point_src[0:2], normalized_point_dst[0:2]]

    return normalized_points


""" 用于特征点匹配，对应矩阵求解的函数 """
def findHomography(src_points, dst_points, h1, w1, h2, w2, threshold=1.0, conf=0.95, max_iters=10000):
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

    # A ← Build neighborhood-graph using r.
    # 初始化 Graph A
    cell_number_in_neighborhood_graph_ = 8
    neighborhood_graph = GridNeighborhoodGraph(points,
                                          w1 / cell_number_in_neighborhood_graph_,
                                          h1 / cell_number_in_neighborhood_graph_,
                                          w2 / cell_number_in_neighborhood_graph_,
                                          h2 / cell_number_in_neighborhood_graph_,
                                          cell_number_in_neighborhood_graph_)
    # 检查是否初始化成功
    if not neighborhood_graph.initialized:
        print("领域图初始化失败\n")
        return None

    ''' GC-RANSAC过程 '''
    # 设置模型估计器和模型
    estimator = EstimatorHomography(SolverHomographyFourPoint, 
                                    SolverHomographyFourPoint)
    model = Homography()

    # 设置全局样本和LO局部优化样本
    main_sampler = ProgressiveNapsacSampler(points,
                                            [16, 8, 4, 2],  # 网格层, 最细网格的单元是有维度的
                                            estimator.sampleSize(),  # 最小样本数目
                                            w1, h1, w2, h2,
                                            sampler_length=0.5)
    local_optimization_sampler = UniformSampler(points) # 局部优化采样器用于局部优化
    # 检查样本是否成功初始化
    if not main_sampler.initialized or not local_optimization_sampler.initialized:
        print("采样器初始化失败\n")
        return None

    # 求解图像最大对角线距离，用于设置图像匹配的阈值
    max_image_diagonal = math.sqrt(max(w1, w2) ** 2 + max(h1, h2) ** 2)
    # 设置GC-RANSAC算法参数
    gcransac = GCRANSAC()
    gcransac.settings.threshold = threshold
    gcransac.settings.spatial_coherence_weight = 0.14
    gcransac.settings.confidence = conf
    gcransac.settings.max_local_optimization_number = 50
    gcransac.settings.max_iteration_number = max_iters
    gcransac.settings.min_iteration_number = 50
    gcransac.settings.neighborhood_sphere_radius = 8
    gcransac.settings.core_number = 4

    # 运行GC-RANSAC算法
    model, inliers = gcransac.run(points,
                                  estimator,
                                  main_sampler,
                                  local_optimization_sampler,
                                  neighborhood_graph)

    print(f'Applied number of local optimizations = {gcransac.statistics.local_optimization_number}')
    print(f'Applied number of graph-cuts = {gcransac.statistics.graph_cut_number}')
    print(f'Applied number of graph-cuts-better = {gcransac.statistics.graph_cut_better_number}')
    print(f'Number of iterations = {gcransac.statistics.iteration_number}')

    # 获取特征点匹配结果（变换矩阵 和 模型对应内点）
    H = model.descriptor
    mask = __transformInliersToMask(inliers, gcransac.point_number)
    return H, mask


def findFundamentalMat(src_points, dst_points, h1, w1, h2, w2, threshold=1.0, conf=0.95, max_iters=10000):
    """ 基础矩阵求解

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

    # A ← Build neighborhood-graph using r.
    # 初始化 Graph A
    cell_number_in_neighborhood_graph_ = 8
    neighborhood_graph = GridNeighborhoodGraph(points,
                                          w1 / cell_number_in_neighborhood_graph_,
                                          h1 / cell_number_in_neighborhood_graph_,
                                          w2 / cell_number_in_neighborhood_graph_,
                                          h2 / cell_number_in_neighborhood_graph_,
                                          cell_number_in_neighborhood_graph_)
    # 检查是否初始化成功
    if not neighborhood_graph.initialized:
        print("领域图初始化失败\n")
        return None

    ''' GC-RANSAC过程 '''
    # 设置模型估计器和模型
    estimator = EstimatorFundamental(SolverFundamentalMatrixEightPoint,
                                     SolverFundamentalMatrixEightPoint,
                                     minimum_inlier_ratio_in_validity_check=0.5)
    model = FundamentalMatrix()

    # 设置全局样本和LO局部优化样本
    main_sampler = ProsacSampler(points, estimator.sampleSize())
    '''
    main_sampler = ProgressiveNapsacSampler(points,
                                            [16, 8, 4, 2],          # 网格层, 最细网格的单元是有维度的
                                            estimator.sampleSize(), # 最小样本数目
                                            w1, h1, w2, h2)
                                            '''
    local_optimization_sampler = UniformSampler(points)             # 局部优化采样器用于局部优化
    # 检查样本是否成功初始化
    if not main_sampler.initialized or not local_optimization_sampler.initialized:
        print("采样器初始化失败\n")
        return None

    # 求解图像最大对角线距离，用于设置图像匹配的阈值
    max_image_diagonal = math.sqrt(max(w1, w2) ** 2 + max(h1, h2) ** 2)
    # 设置GC-RANSAC算法参数
    gcransac = GCRANSAC()
    gcransac.settings.threshold = 0.0003 * threshold * max_image_diagonal
    gcransac.settings.spatial_coherence_weight = 0.14
    gcransac.settings.confidence = conf
    gcransac.settings.max_local_optimization_number = 50
    gcransac.settings.max_iteration_number = max_iters
    gcransac.settings.min_iteration_number = 50
    gcransac.settings.neighborhood_sphere_radius = 8
    gcransac.settings.core_number = 4

    # 运行GC-RANSAC算法
    model, inliers = gcransac.run(points,
                                  estimator,
                                  main_sampler,
                                  local_optimization_sampler,
                                  neighborhood_graph)

    print(f'Applied number of local optimizations = {gcransac.statistics.local_optimization_number}')
    print(f'Applied number of graph-cuts = {gcransac.statistics.graph_cut_number}')
    print(f'Applied number of graph-cuts-better = {gcransac.statistics.graph_cut_better_number}')
    print(f'Number of iterations = {gcransac.statistics.iteration_number}')

    # 获取GC-RANSAC结果（变换矩阵 和 模型对应内点）
    F = model.descriptor
    mask = __transformInliersToMask(inliers, gcransac.point_number)
    return F, mask


def findEssentialMat(src_points, dst_points, src_K, dst_K, h1, w1, h2, w2, threshold=1.0, conf = 0.95, max_iters = 10000):
    """ 基础矩阵求解

    参数
    --------
    src_points : numpy
        源图像特征点集合
    dst_points : numpy
        目标图像特征点集合
    src_K : numpy
        源图像相机内参矩阵
    dst_K : numpy
        目标图像相机内参矩阵
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

    # A ← Build neighborhood-graph using r.
    # 初始化 Graph A
    cell_number_in_neighborhood_graph = 8
    neighborhood_graph = GridNeighborhoodGraph(points,
                                          w1 / cell_number_in_neighborhood_graph,
                                          h1 / cell_number_in_neighborhood_graph,
                                          w2 / cell_number_in_neighborhood_graph,
                                          h2 / cell_number_in_neighborhood_graph,
                                          cell_number_in_neighborhood_graph)
    # 检查是否初始化成功
    if not neighborhood_graph.initialized:
        print("领域图初始化失败\n")
        return None

    # 求解由内参矩阵转换后的点集
    normalized_points = __normalizeCorrespondences(points, src_K, dst_K)

    ''' GC-RANSAC过程 '''
    # 设置模型估计器和模型
    estimator = EstimatorEssential(SolverEssentialMatrixEightPoint,
                                   SolverEssentialMatrixEightPoint,
                                   src_K, dst_K,
                                   minimum_inlier_ratio_in_validity_check=0.3)
    model = EssentialMatrix()

    # 设置全局样本和LO局部优化样本
    main_sampler = ProgressiveNapsacSampler(points,
                                            [16, 8, 4, 2],          # 网格层, 最细网格的单元是有维度的
                                            estimator.sampleSize(), # 最小样本数目
                                            w1, h1, w2, h2)
    local_optimization_sampler = UniformSampler(points)             # 局部优化采样器用于局部优化
    # 检查样本是否成功初始化
    if not main_sampler.initialized or not local_optimization_sampler.initialized:
        print("采样器初始化失败\n")
        return None

    # 求解图像最大对角线距离，用于设置图像匹配的阈值
    max_image_diagonal = math.sqrt(max(w1, w2) ** 2 + max(h1, h2) ** 2)
    # 设置GC-RANSAC算法参数
    gcransac = GCRANSAC()
    gcransac.settings.threshold = 0.0003 * threshold
    gcransac.settings.spatial_coherence_weight = 0.14
    gcransac.settings.confidence = conf
    gcransac.settings.max_local_optimization_number = 50
    gcransac.settings.max_iteration_number = max_iters
    gcransac.settings.min_iteration_number = 50
    gcransac.settings.neighborhood_sphere_radius = 8
    gcransac.settings.min_iteration_number_before_lo = 5
    gcransac.settings.core_number = 4

    # 运行GC-RANSAC算法
    model, inliers = gcransac.run(normalized_points,
                                  estimator,
                                  main_sampler,
                                  local_optimization_sampler,
                                  neighborhood_graph)

    print(f'Applied number of local optimizations = {gcransac.statistics.local_optimization_number}')
    print(f'Applied number of graph-cuts = {gcransac.statistics.graph_cut_number}')
    print(f'Applied number of graph-cuts-better = {gcransac.statistics.graph_cut_better_number}')
    print(f'Number of iterations = {gcransac.statistics.iteration_number}')

    # 获取GC-RANSAC结果（变换矩阵 和 模型对应内点）
    E = model.descriptor
    mask = __transformInliersToMask(inliers, gcransac.point_number)
    return E, mask
