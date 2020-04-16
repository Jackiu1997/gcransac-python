import math as m
import sys

import cv2
import numpy as np

from graph import GraphCut
from model import *
from neighbor.grid_neighborhood_graph import GridNeighborhoodGraph
from utils.score import MSACScoringFunction, Score


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
        self.graph_cut_number = 0
        self.local_optimization_number = 0
        self.iteration_number = 0
        self.neighbor_number = 0

class GCRANSAC:

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
        self.scoring_function = MSACScoringFunction()

        # GrpahCut 最小化优化求解器
        self.energy = GraphCut()

    def run(self,
            points,
            estimator,
            main_sampler,
            local_optimization_sampler,
            neighborhood_graph):
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
        self.neighborhood_graph = neighborhood_graph
        self.main_sampler = main_sampler
        self.local_optimization_sampler = local_optimization_sampler
        self.step_size = self.point_number / self.settings.core_number

        # 最小二乘法截断误差，模型评分函数初始化
        self.truncated_threshold = 3.0 / 2.0 * self.settings.threshold
        self.scoring_function.initialize(self.truncated_threshold ** 2, self.point_number)

        ''' The main RANSAC iteration '''

        # 记录全局的最佳模型，得分，内点集合
        so_far_the_best_model = Model()
        so_far_the_best_score = Score()
        last_the_best_score = Score()
        so_far_the_best_inliers = []

        # 初始化采样池
        pool = [i for i in range(self.point_number)]

        # init H(|L∗|, µ)
        self.max_iteration = self.__getIterationNumber(1)

        # for k = 1 →H(|L∗|, µ) do
        while self.statistics.iteration_number < min(self.max_iteration, self.settings.max_iteration_number):
            do_local_optimization = False
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
                                                                self.settings.threshold,
                                                                so_far_the_best_score)
                '''
                # 检查模型是否有效，无效则重新评估
                result = self.estimator.isValidModel(model,
                                                     data=points,
                                                     inliers=inliers,
                                                     minimal_sample=sample,
                                                     threshold=self.settings.threshold)
                if isinstance(result, tuple):
                    model = result[1]
                elif not result:
                    continue
                '''

                # if wk > w∗ then
                # 	θ∗, L∗, w∗ ← θk, Lk, wk
                if so_far_the_best_score < score:
                    last_the_best_score = so_far_the_best_score
                    so_far_the_best_model = model
                    so_far_the_best_score = score
                    so_far_the_best_inliers = inliers
                    # 更新最大迭代数
                    self.max_iteration = self.__getIterationNumber(so_far_the_best_score.inlier_number)

                    # µ21 = µ2/µ1 < lo_conf
                    # 决定是否需要局部优化
                    if last_the_best_score.inlier_number != 0:
                        u2 = self.__getConfidenceNumber(so_far_the_best_score.inlier_number)
                        u1 = self.__getConfidenceNumber(last_the_best_score.inlier_number)
                        u21 = u2 / u1
                        do_local_optimization = True if u21 > 1.3 else do_local_optimization
            
            # if do_local_optimization then
            if do_local_optimization and self.settings.do_local_optimization:
                # θLO, LLO, wLO ← Local opt.
                result = self.__localOptimization(so_far_the_best_model,
                                                  so_far_the_best_inliers,
                                                  so_far_the_best_score)

                # if wLO > w∗ then
                # 	θ∗, L∗, w∗ ← θLO, LLO, wLO
                if result != None and so_far_the_best_score < result[2]:
                    so_far_the_best_model = result[0]
                    so_far_the_best_inliers = result[1]
                    so_far_the_best_score = result[2]
                    # 更新最大迭代数
                    self.max_iteration = self.__getIterationNumber(so_far_the_best_score.inlier_number)

        # if nLO = 0 then
        if self.statistics.local_optimization_number == 0 and self.settings.do_local_optimization:
            # θ∗, L∗, w∗ ← Local opt.
            result = self.__localOptimization(so_far_the_best_model,
                                              so_far_the_best_inliers,
                                              so_far_the_best_score)
            if result != None and so_far_the_best_score < result[2]:
                so_far_the_best_model = result[0]
                so_far_the_best_inliers = result[1]
                so_far_the_best_score = result[2]
        
        # θ∗ ← least squares model fitting using L∗.
        if self.settings.do_final_iterated_least_squares:
            result = self.__iteratedLeastSquaresFitting(so_far_the_best_model,
                                                        so_far_the_best_inliers,
                                                        so_far_the_best_score)
            # θ∗, L∗, w∗ ← least squares fitting.
            if result != None and so_far_the_best_score < result[2]:
                so_far_the_best_model = result[0]
                so_far_the_best_inliers = result[1]
                so_far_the_best_score = result[2]

        # Output: θ - model parameters; L – labeling
        return so_far_the_best_model, so_far_the_best_inliers

    def __localOptimization(self,
                            so_far_the_best_model,
                             so_far_the_best_inliers,
                             so_far_the_best_score):
        """ 通过图割算法确定内点的局部优化估测 
        
        参数
        ----------
        so_far_the_best_model : Model
            最佳模型参数
        so_far_the_best_inliers : list
            最佳内点序号列表
        so_far_the_best_score : Score
            最佳模型评估得分

        返回
        ----------
        Model, list, Score
            局部优化最佳模型，最佳内点，最佳得分
        """
        if self.statistics.graph_cut_number >= self.settings.max_graph_cut_number:
            return None
        self.statistics.graph_cut_number += 1
        self.statistics.local_optimization_number += 1

        # G ← Build the problem graph.
        # L ← Apply graph-cut to G.
        # I7m ← Select a 7m-sized random inlier set.
        gc_inliers = self.energy.labeling(self.points,
                                          so_far_the_best_model,
                                          self.estimator,
                                          self.neighborhood_graph,
                                          self.settings.spatial_coherence_weight,
                                          self.settings.threshold)

        # θ ← Fit a model using labeling I7m
        sample_size = min(self.estimator.inlierLimit(), len(gc_inliers))
        changed = False
        unsuccess_iteration = 0
        while not changed and unsuccess_iteration < 10:
            unsuccess_iteration += 1

            # 采样数小于内点数，则均匀随机采样，用 I7m 样本估计模型
            if sample_size < len(gc_inliers):
                current_sample = self.local_optimization_sampler.sample(gc_inliers, sample_size)
            # 模型估计器所需采样数小于内点数，则用内点之间估计模型
            elif self.estimator.sampleSize() < len(gc_inliers):
                current_sample = gc_inliers
            # 否则，内点数目不能估计模型
            else:
                break

            models = self.estimator.estimateModelNonminimal(self.points,
                                                            current_sample,
                                                            sample_size)
            if len(models) == 0:
                continue
            
            for model in models:
                # w ← Compute the support of θ
                score, inliers = self.scoring_function.getScore(self.points,
                                                                model,
                                                                self.estimator,
                                                                self.settings.threshold,
                                                                so_far_the_best_score)
                # f w > w∗LO then
                # 	θ∗LO, L∗LO, w∗LO, changed ← θ, L, w, 1.
                if so_far_the_best_score < score:
                    so_far_the_best_score = score
                    so_far_the_best_model = model
                    so_far_the_best_inliers = inliers
                    changed = True

        # Output: L∗LO – labeling, w∗LO – support, θ∗LO – model
        return so_far_the_best_model, so_far_the_best_inliers, so_far_the_best_score

    def __iteratedLeastSquaresFitting(self,
                                      so_far_the_best_model,
                                      so_far_the_best_inliers,
                                      so_far_the_best_score,
                                      use_weighting=True):
        """ 通过最小二乘法拟合模型 
        
        参数
        ----------
        so_far_the_best_model : Model
            最佳模型参数
        so_far_the_best_inliers : list
            最佳内点序号列表
        so_far_the_best_score : Score
            最佳模型评估得分
        use_weighting : bool
            是否使用加权进行最小二乘拟合模型

        返回
        ----------
        Model, list, Score
            最小二乘拟合最佳模型，最佳内点，最佳得分
        """
        sample_size = self.estimator.sampleSize()
        if len(so_far_the_best_inliers) <= sample_size:
            return None

        # 最小二乘法拟合模型
        squared_threshold = self.truncated_threshold ** 2
        weights = [1.0 for i in range(self.point_number)]
        iterations = 0
        while iterations+1 < self.settings.max_least_squares_iterations:
            iterations += 1
            
            # 如果有权重，则输入权重，否则输入 None
            if self.estimator.isWeightingApplicable() and use_weighting:
                for point_idx in so_far_the_best_inliers:
                    squared_residual = self.estimator.squaredResidual(self.points[point_idx], so_far_the_best_model)
                    weight = max(0.0, 1.0 - squared_residual / squared_threshold)
                    weights[point_idx] = weight ** 2
            # 通过内点和权重估计模型
            models = self.estimator.estimateModelNonminimal(self.points, 
                                                            so_far_the_best_inliers, 
                                                            len(so_far_the_best_inliers), 
                                                            weights=weights)
            if len(models) == 0:
                break

            updated = False
            for model in models:
                # 计算当前模型的得分和内点集合
                score, inliers = self.scoring_function.getScore(self.points,
                                                                model,
                                                                self.estimator,
                                                                self.truncated_threshold,
                                                                so_far_the_best_score)
                # 获取内点数目小于最小样本数，则继续
                if len(inliers) < sample_size:
                    continue
                # 如果内点数目没有更改，则继续
                if score.inlier_number <= len(so_far_the_best_inliers):
                    continue
                # 如果模型内点更多，更新模型和内点集
                if score.inlier_number >= so_far_the_best_score.inlier_number:
                    so_far_the_best_model = model
                    so_far_the_best_inliers = inliers
                    so_far_the_best_score = score
                    updated = True

            # 如果模型未被更新，则中断程序
            if not updated:
                break

        return so_far_the_best_model, so_far_the_best_inliers, so_far_the_best_score
    
    # H(|L∗|, µ)
    def __getIterationNumber(self, inlier_number):
        """ 计算当前内点数目期望的迭代数目 """
        Pi = (inlier_number / self.point_number) ** self.estimator.sampleSize()
        if Pi < sys.float_info.epsilon:
            return sys.maxsize
        log1 = m.log(1- self.settings.confidence)
        log2 = m.log(1.0 - Pi)
        return int(log1 / log2) + 1

    def __getConfidenceNumber(self, inlier_number):
        """ 计算当前模型的内点置信概率 """
        inlier_ratio = float(inlier_number) / self.point_number  # η
        # µ = 1 − 10^k * log(1 − η^m)
        confidenceNumber = 1 - \
                10 ** (self.max_iteration *
                    m.log(1 - inlier_ratio ** self.estimator.sampleSize()))
        return confidenceNumber
