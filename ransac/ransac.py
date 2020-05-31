import math as m
import sys

import cv2
import numpy as np

from graph import GraphCut
from model import *
from neighbor.grid_neighborhood_graph import GridNeighborhoodGraph
from utils.score import RansacScoringFunction


class _Settings:
    
    def __init__(self):
        self.do_final_iterated_least_squares = True      # 是否执行最后的最小二乘拟合模型
        self.do_local_optimization = True                # 是否需要局部优化拟合模型
        self.do_graph_cut = True                         # 是否执行图割算法
        self.use_inlier_limit = False                    # 是否确定内点限制以加速局部优化过程

        self.min_iteration_number = 20                   # 全局最小迭代次数
        self.max_iteration_number = 32767                # 全局最大迭代次数
        self.min_iteration_number_before_lo = 20         # 执行局部优化前最小迭代次数
        self.max_local_optimization_number = 10          # 局部优化最大迭代次数
        self.max_least_squares_iterations = 10           # 最小二乘法拟合的最大迭代次数
        self.max_graph_cut_number = 10                   # 图割算法求解内点最大使用次数
        self.core_number = 1                             # CPU 核心数目

        self.confidence = 0.95                           # 结果的置信率
        self.threshold = 2.0                             # 决定内点和外点的阈值
        self.neighborhood_sphere_radius = 20             # 构建领域图的区域半径
        self.spatial_coherence_weight = 0.14             # 空间相干性能量权重


class _Statistics:

    def __init__(self):
        self.iteration_number = 0
        self.graph_cut_number = 0
        self.graph_cut_better_number = 0
        self.local_optimization_number = 0
        self.neighbor_number = 0
        

class RANSAC:

    def __init__(self):
        # 设置初始化
        self.settings = _Settings()
        self.statistics = _Statistics()

        # 设置 estimator 和 gridgraph
        self.estimator = None
        self.neighborhood_graph = None

        # 设置一系列参数
        self.max_iteration = 0
        self.points = None
        self.point_number = 0
        self.sample_number = 0
        self.truncated_threshold = 0.0          # 3 / 2 * threshold_
        self.squared_truncated_threshold = 0.0	
        self.step_size = 1                      # 每个进程的步数

        # 全局采样器和局部采样器
        self.main_sampler = None
        self.local_optimization_sampler = None

        # 模型评估的评分函数
        self.scoring_function = RansacScoringFunction()

    def run(self,
            points,
            estimator,
            main_sampler):
        """ 运行 GC-RANSAC 求解过程
        
        参数
        ----------
        points : numpy
            输入的点集合
        estimator : Estimator
            模型的估计器
        main_sampler : Sampler
            全局采样器
        local_optimization_sampler : Sampler
            局部优化采样器
        neighborhood_graph : GridNeighborGraph
            构建的领域图

        返回
        ----------
        Model, list(int)
            求解的最佳模型和内点序号列表
        """
        # 初始化参数赋值
        self.points = points
        self.point_number = np.shape(points)[0]
        self.sample_number = estimator.sampleSize()

        self.estimator = estimator
        self.main_sampler = main_sampler

        ''' The main RANSAC iteration '''

        # 记录全局的最佳模型，得分，内点集合
        so_far_the_best_model = Model()
        so_far_the_best_score = 0
        so_far_the_best_inliers = []

        # 初始化采样池
        pool = [i for i in range(self.point_number)]

        # init H(|L∗|, µ)
        self.max_iteration = self.__getIterationNumber(1)

        # for k = 1 →H(|L∗|, µ) do
        while self.statistics.iteration_number < min(self.max_iteration, self.settings.max_iteration_number):
            # 增加迭代计算次数
            self.statistics.iteration_number += 1

            # Sk ← Draw a minimal sample
            sample = self.main_sampler.sample(pool, self.sample_number)
            # 检查采样是否有效，无效则重新评估
            if len(sample) == 0 or\
                not self.estimator.isValidSample(points, sample):
                continue

            # θk ← Estimate a model using Sk
            models = self.estimator.estimateModel(points, sample)
            if len(models) == 0:
                continue

            for model in models:
                # wk ← Compute the support of θk
                score, inliers = self.scoring_function.getScore(points,
                                                                model,
                                                                self.estimator,
                                                                self.settings.threshold)
                # 检查模型是否有效，无效则重新评估
                if not self.estimator.isValidModel(model,
                                                   data=points,
                                                   inliers=inliers,
                                                   minimal_sample=sample,
                                                   threshold=self.settings.threshold):
                    continue

                # if wk > w∗ then
                # 	θ∗, L∗, w∗ ← θk, Lk, wk
                if so_far_the_best_score < score:
                    last_the_best_score = so_far_the_best_score
                    so_far_the_best_model = model
                    so_far_the_best_score = score
                    so_far_the_best_inliers = inliers
                    # 更新最大迭代数
                    self.max_iteration = self.__getIterationNumber(score)

        # Output: θ - model parameters; L – labeling
        return so_far_the_best_model, so_far_the_best_inliers

    # H(|L∗|, µ)
    def __getIterationNumber(self, inlier_number):
        """ 计算当前内点数目期望的迭代数目 """
        inlier_ratio = float(inlier_number) / self.point_number  # η
        if inlier_ratio ** self.estimator.sampleSize() < sys.float_info.epsilon:
            return sys.maxsize
        log1 = m.log(1- self.settings.confidence)
        log2 = m.log(1.0 - inlier_ratio ** self.estimator.sampleSize())
        return int(log1 / log2) + 1