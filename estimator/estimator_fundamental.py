import math as m
import sys

import numpy as np
import numpy.linalg as linalg

from gcransac.gcransac import GCRANSAC
from model import *
from sampler import UniformSampler
from solver import (SolverFundamentalMatrixEightPoint,
                    SolverFundamentalMatrixPlaneParallax,
                    SolverFundamentalMatrixSevenPoint,
                    SolverHomographyFourPoint)

from .estimator import Estimator
from .estimator_homography import EstimatorHomography


class EstimatorFundamental(Estimator):
    """ 基础矩阵估计器 """

    def __init__(self,
                 minimalSolver,
                 nonMinimalSolver,
                 minimum_inlier_ratio_in_validity_check=0.5):
        super().__init__()
        # 用于估计最小样本模型的估计器
        self.minimal_solver = minimalSolver()
        # 用于估计非最小样本模型的估计器
        self.non_minimal_solver = minimalSolver()
        # 通过有效性测试所需的内点比率的下限
        self.minimum_inlier_ratio_in_validity_check = minimum_inlier_ratio_in_validity_check

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
        models = self.minimal_solver.estimateModel(data,
                                                   sample,
                                                   sample_size)
        # 对极约束检验 
        for model in models:
            if not self.__isOrientationValid(model.descriptor,
                                             data,
                                             sample,
                                             sample_size):
                models.remove(model)
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
        models = self.non_minimal_solver.estimateModel(data,
                                                       sample,
                                                       sample_number,
                                                       weights=weights)
        # 对极约束检验 
        for model in models:
            if not self.__isOrientationValid(model.descriptor,
                                             data,
                                             sample,
                                             sample_number):
                models.remove(model)
        return models
    
    def residual(self, point, model):
        ''' 给定模型和数据点，计算误差 '''
        return m.sqrt(self.__sampsonDistance(point, model.descriptor))

    def squaredResidual(self, point, model):
        """ 给定模型和数据点，计算误差的平方 """
        return self.__sampsonDistance(point, model.descriptor)

    ''' 检查模型是否有效 '''
    def isValidModel(self,
                     model,
                     data=None,
                     inliers=None,
                     minimal_sample=None,
                     threshold=None):
        """ 检查模型是否有效，可以是模型结构的几何检查或其他验证
        通过检查具有对称外极距的内点数验证模型, 而不是 sampson 距离。一般来说，Sampson距离更准确，但不太准确
        的对称极距对退化解更具鲁棒性。因此，到目前为止，最好的模型都会被检查是否有足够的内部对称的

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
        return True
        # 当使用对称极距而不是 Sampson 距离时，也应该是内点的最小数
        minimum_inlier_number = max(self.sampleSize(), len(inliers) * self.minimum_inlier_ratio_in_validity_check)
        inlier_number = 0
        descriptor = model.descriptor
        squared_threshold = threshold ** 2

        # 遍历由 sampson 距离确定的内点
        for idx in inliers:
            # 计算对称极距，并确定内点数目（如果内点数大于最小内点，则模型通过）
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

        return x2_f_x1 ** 2 * (1 / (f_x1[0] ** 2 + f_x1[1] ** 2) + 1 / (x2_f[0] ** 2 + x2_f[1] ** 2))\

    ''' 对极约束函数 Oriented epipolar constraints '''
    def __getEpipole(self, fundamental_matrix):
        epsilon = sys.float_info.epsilon
        epipole =  np.cross(fundamental_matrix[0], fundamental_matrix[2])

        for i in range(3):
            if (epipole[i] > epsilon) or (epipole[i] < -epsilon):
                return epipole

        epipole = np.cross(fundamental_matrix[1], fundamental_matrix[2])
        return epipole

    def __getOrientationSignum(self, fundamental_matrix, epipole, point):
        signum1 = fundamental_matrix[0, 0] * point[2] + fundamental_matrix[1, 0] * point[3] + fundamental_matrix[2, 0]
        signum2 = epipole[1] - epipole[2] * point[1]
        return signum1 * signum2

    def __isOrientationValid(self, fundamental_matrix, data, sample, sample_size):
        """ 检查朝向约束是否有效 
        
        参数
        ---------
        fundamental_matrix : numpy
            基础矩阵
        data : numpy
            数据点集合
        sample : list
            样本点序号列表
        sample_size : int
            样本点数目
        
        返回
        ---------
        bool
            朝向约束是有效
        """
        epipole = self.__getEpipole(fundamental_matrix)

        if sample == None:
            sample = [i for i in range(sample_size)]
        # 获取样本中第一个点的方向标志
        signum2 = self.__getOrientationSignum(fundamental_matrix, epipole, data[sample[0]])
        for i in range(sample_size):
            # 获取样本中第 i 个点的方向标志
            signum1 = self.__getOrientationSignum(fundamental_matrix, epipole, data[sample[i]])
            # 符号应该相等，否则，基本矩阵无效
            if signum2 * signum1 < 0:
                return False

        return True