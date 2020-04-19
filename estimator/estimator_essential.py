import math as m

import numpy as np

from model import EssentialMatrix
from solver import SolverEssentialMatrixFivePointStewenius

from .estimator import Estimator
import sys


class EstimatorEssential(Estimator):
    """ 本质矩阵估计器 """

    def __init__(self,
                 minimalSolver,
                 nonMinimalSolver,
                 intrinsics_src,
                 intrinsics_dst,
                 minimum_inlier_ratio_in_validity_check=0.5,
                 point_ratio_for_selecting_from_multiple_models=0.05):
        super().__init__()
        # 用于估计最小样本模型的估计器
        self.minimal_solver = minimalSolver()
        # 用于估计非最小样本模型的估计器
        self.non_minimal_solver = nonMinimalSolver()

        self.intrinsics_src = intrinsics_src # 源图像相机内参
        self.intrinsics_dst = intrinsics_dst # 目标图像相机内参

        # 通过有效性测试所需的内点比率的下限
        self.minimum_inlier_ratio_in_validity_check = minimum_inlier_ratio_in_validity_check
        # 当非最小模型拟合方法返回多个模型时使用的点的比率
        self.point_ratio_for_selecting_from_multiple_models = point_ratio_for_selecting_from_multiple_models

    def sampleSize(self):
        """ 估计模型所需的最小样本的大小 """
        return self.minimal_solver.sampleSize()

    def nonMinimalSampleSize(self):
        """ 估计模型所需的非最小样本的大小 """
        return self.non_minimal_solver.sampleSize()

    ''' 模型估计函数 '''
    def estimateModel(self, data, sample):
        """ 给定一组数据点，估计最小样本模型
        
        参数
        ----------
        data : numpy
            输入的数据点集
        sample : list
            用于估计模型的样本点序号列表

        返回
        ----------
        list(Model)
            通过样本估计的模型列表
        """
        sample_size = self.sampleSize()
        models = self.minimal_solver.estimateModel(data, sample, sample_size)
        return models

    def estimateModelNonminimal(self, data, sample, sample_number, weights=None):
        """ 根据数据点集的非最小采样估计模型

        参数
        ----------
        data : numpy
            输入的数据点集
        sample : list
            用于估计模型的样本点序号列表
        sample_number : int
            样本点数目
        weights : list
            数据点集中点的对应权重

        返回
        ----------
        list(Model)
            通过样本估计的模型列表
        """
        if sample_number < self.nonMinimalSampleSize():
            return []

        # 用于从估计的模型中选择最佳模型的点数
        points_not_used = max(1, round(sample_number * self.point_ratio_for_selecting_from_multiple_models)
                              ) if self.non_minimal_solver.returnMultipleModels() else 0
        points_used = sample_number - points_not_used

        models = self.non_minimal_solver.estimateModel(data,
                                                       sample,
                                                       points_used,
                                                       weights=weights)
        return models

    ''' 给定模型和数据点，计算误差 '''
    def residual(self, point, model):
        return m.sqrt(self.__sampsonDistance(point, model.descriptor))

    def squaredResidual(self, point, model):
        return self.__sampsonDistance(point, model.descriptor)

    ''' 检查模型是否有效 '''
    def isValidModel(self,
                     model,
                     data=None,
                     inliers=None,
                     minimal_sample=None,
                     threshold=None):
        """ 检查模型是否有效，本质矩阵模型的对称极距的校验

        参数
        ----------
        model : Model
            需要检查的模型
        data : numpy
            输入的数据点集
        inliers : list
            需要检查的模型的内点
        minimal_sample : int
            样本点的数目
        threshold : float
            决定内点和外点的阈值

        返回
        ----------
        bool
            模型是否有效
        """
        # 当使用对称极距而不是 Sampson 距离时，也应该是最小内点数
        minimum_inlier_number = max(self.sampleSize(), len(inliers) * self.minimum_inlier_ratio_in_validity_check)
        inlier_number = 0
        descriptor = model.descriptor
        squared_threshold = threshold ** 2
        
        # 通过 sampson 距离计算确定内点数目
        for idx in inliers:
            # 使用对称极距计算并检查是否小于阈值
            if self.__symmetricEpipolarDistance(data[idx], descriptor) < squared_threshold:
                inlier_number += 1
                if inlier_number >= minimum_inlier_number:
                    return True
        return False

    ''' 距离计算工具函数 '''
    def __sampsonDistance(self, point, descriptor):
        """ 点对应与本质矩阵的 sampson 距离 """
        x1 = np.hstack((point[0:2], [1]))
        x2 = np.hstack((point[2:4], [1]))
        
        f_x1 = np.dot(descriptor, x1)
        x2_f = np.dot(x2.T, descriptor)
        x2_f_x1 = np.dot(x2_f, x1)

        return x2_f_x1 ** 2 / (f_x1[0] ** 2 + f_x1[1] ** 2 + x2_f[0] ** 2 + x2_f[1] ** 2)

    def __symmetricEpipolarDistance(self, point, descriptor):
        """ 点对应与本质矩阵的 对称极线距离 """
        x1 = np.hstack((point[0:2], [1]))
        x2 = np.hstack((point[2:4], [1]))
        
        f_x1 = np.dot(descriptor, x1)
        x2_f = np.dot(x2.T, descriptor)
        x2_f_x1 = np.dot(x2_f, x1)

        return x2_f_x1 ** 2 * (1 / (f_x1[0] ** 2 + f_x1[1] ** 2) + 1 / (x2_f[0] ** 2 + x2_f[1] ** 2))