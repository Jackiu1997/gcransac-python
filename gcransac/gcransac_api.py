import math

import cv2
import numpy as np

from estimator import EstimatorHomography
from model import *
from neighbor import GridNeighborhoodGraph
from sampler import ProgressiveNapsacSampler, UniformSampler
from solver.solver_homography_four_point import SolverHomographyFourPoint

from .gcransac import GCRANSAC


# 对外过程的api，用于对应矩阵的求解
# 单应矩阵求解
def findHomography(src_points, dst_points, h1, w1, h2, w2, threshold):
    # 初始化默认参数
    threshold = 1.0
    conf = 0.99
    max_iters = 10000

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
                                            sampler_length=0.5)  # 完全混合到全局采样的长度（即 0.5*<point number> 迭代次数）
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
    gcransac.settings.max_iteration_number = 5000
    gcransac.settings.min_iteration_number = 50
    gcransac.settings.neighborhood_sphere_radius = 8
    gcransac.settings.core_number = 4

    # 运行GC-RANSAC算法
    model, inliers = gcransac.run(points,
                                  estimator,
                                  main_sampler,
                                  local_optimization_sampler,
                                  neighborhood_graph)

    print("iter num\t", gcransac.statistics.iteration_number)
    print("lo num\t", gcransac.statistics.local_optimization_number)
    print("gc num\t", gcransac.statistics.graph_cut_number)

    # 获取特征点匹配结果（变换矩阵 和 模型对应内点）
    H = model.descriptor
    mask = []
    for i in range(np.shape(points)[0]):
        if i in inliers:
            mask.append(1)
        else:
            mask.append(0)
    mask = np.array(mask)
    return H, mask
