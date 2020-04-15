import math as m

import numpy as np
import numpy.linalg as linalg

from model import *
from .estimator import Estimator
from solver.solver_homography_four_point import SolverHomographyFourPoint


class EstimatorHomography(Estimator):
    """ 单应矩阵估计器 """

    def __init__(self, minimalSolver, nonMinimalSolver):
        super().__init__()
        # 用于估计最小样本模型的估计器
        self.minimal_solver = minimalSolver()
        # 用于估计非最小样本模型的估计器
        self.non_minimal_solver = minimalSolver()

    def sampleSize(self):
        """ 估计模型所需的最小样本的大小 """
        return self.minimal_solver.sampleSize()

    def nonMinimalSampleSize(self):
        """ 估计模型所需的非最小样本的大小 """
        return self.non_minimal_solver.sampleSize()

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
        models = self.minimal_solver.estimateModel(data,
                                                   sample,
                                                   self.sampleSize())
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

        # 在应用最小二乘模型拟合时，对点坐标进行归一化以实现数值稳定性
        normalized_points, normalizing_transform_source, normalizing_transform_destination = self.__normalizePoints(
            data, sample, sample_number)

        models = self.non_minimal_solver.estimateModel(normalized_points,
                                                       None,
                                                       sample_number,
                                                       weights=weights)
        # 估计基本矩阵的反归一化
        for model in models:
            model.descriptor = np.dot(np.linalg.inv(normalizing_transform_destination), model.descriptor)
            model.descriptor = np.dot(model.descriptor, normalizing_transform_source)
        return models

    def residual(self, point, model):
        """ 给定模型和数据点，计算误差 """
        return m.sqrt(squaredResidual(point, model))

    def squaredResidual(self, point, model):
        """ 给定模型和数据点，计算误差的平方 """
        descriptor = model.descriptor
        # 计算通过模型变换矩阵后的点坐标
        x1 = point[0]
        y1 = point[1]
        x2 = point[2]
        y2 = point[3]
        t1 = descriptor[0, 0] * x1 + descriptor[0, 1] * y1 + descriptor[0, 2]
        t2 = descriptor[1, 0] * x1 + descriptor[1, 1] * y1 + descriptor[1, 2]
        t3 = descriptor[2, 0] * x1 + descriptor[2, 1] * y1 + descriptor[2, 2]
        # 计算源点转换后与目标点的距离，即为点到模型的距离
        return (x2 - (t1 / t3)) ** 2 + (y2 - (t2 / t3)) ** 2
    
    def isValidSample(self, data, sample):
        """ 在计算模型参数之前判断所选样本是否退化

        参数
        ----------
        data : numpy
            输入的数据点集
        sample : list
            用于估计模型的样本点序号列表

        返回
        ----------
        bool
            样本是否有效
        """
        sample_size = self.sampleSize()

        # 检查朝向约束，取前四个样本点进行交叉验证
        a = data[sample[0]]
        b = data[sample[1]]
        c = data[sample[2]]
        d = data[sample[3]]

        p = self.__cross_product(a[0:2], b[0:2], 1)
        q = self.__cross_product(a[2:4], b[2:4], 1)
        if (p[0] * c[0] + p[1] * c[1] + p[2]) * (q[0] * c[2] + q[1] * c[3] + q[2]) < 0:
            return False
        if (p[0] * d[0] + p[1] * d[1] + p[2]) * (q[0] * d[2] + q[1] * d[3] + q[2]) < 0:
            return False

        p = self.__cross_product(c[0:2], d[0:2], 1)
        q = self.__cross_product(c[2:4], d[2:4], 1)
        if (p[0] * a[0] + p[1] * a[1] + p[2]) * (q[0] * a[2] + q[1] * a[3] + q[2]) < 0:
            return False
        if (p[0] * b[0] + p[1] * b[1] + p[2]) * (q[0] * b[2] + q[1] * b[3] + q[2]) < 0:
            return False

        return True

    def __cross_product(self, vector1, vector2, st):
        """ 计算两个向量的 cross-product """
        result = np.zeros(3)
        result[0] = vector1[st] - vector2[st]
        result[1] = vector2[0] - vector1[0]
        result[2] = vector1[0] * vector2[st] - vector1[st] * vector2[0]
        return result
    
    def __normalizePoints(self, data, sample, sample_number):
        ''' 规范化点集函数 '''
        # 初始化质点坐标
        mass_point_src = np.zeros(2)  # 第一张图片质点
        mass_point_dst = np.zeros(2)  # 第二张图片质点

        # 计算质点坐标 均值
        for i in range(sample_number):
            cur_point = data[sample[i], :]
            # 将坐标添加到质点的坐标上
            mass_point_src += cur_point[0:2]
            mass_point_dst += cur_point[2:4]
        mass_point_src /= sample_number
        mass_point_dst /= sample_number

        # 求解图像点离质点的平均距离
        average_distance_src = 0.0
        average_distance_dst = 0.0
        for i in range(sample_number):
            cur_point = data[sample[i], :]
            # 求解距离
            d1 = mass_point_src - cur_point[0:2]
            d2 = mass_point_dst - cur_point[2:4]
            average_distance_src += m.sqrt(np.sum(d1 ** 2))
            average_distance_dst += m.sqrt(np.sum(d2 ** 2))
        average_distance_src /= sample_number
        average_distance_dst /= sample_number

        # 计算 sqrt（2）/ 平均距离 的比率
        ratio_src = m.sqrt(2) / average_distance_src
        ratio_dst = m.sqrt(2) / average_distance_dst

        # 计算归一化的坐标
        normalized_points_ = np.zeros([sample_number, 4])
        for i in range(sample_number):
            cur_point = data[sample[i], :]
            np1 = (cur_point[0:2] - mass_point_src) * ratio_src
            np2 = (cur_point[2:4] - mass_point_dst) * ratio_dst
            normalized_points_[i] = np.r_[np1, np2]

        # 创建归一化转换
        normalizing_transform_source_ = np.array([[ratio_src, 0, -ratio_src * mass_point_src[0]],
                                                  [0, ratio_src, -ratio_src * mass_point_src[1]],
                                                  [0, 0, 1]])

        normalizing_transform_destination_ = np.array([[ratio_dst, 0, -ratio_dst * mass_point_dst[0]],
                                                       [0, ratio_dst, -ratio_dst * mass_point_dst[1]],
                                                       [0, 0, 1]])
        # 返回归一化坐标，源图像转换矩阵，目标图像转换矩阵
        return normalized_points_, normalizing_transform_source_, normalizing_transform_destination_
