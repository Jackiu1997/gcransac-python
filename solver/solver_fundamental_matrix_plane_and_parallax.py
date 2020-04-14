import sys

import numpy as np
from numpy import linalg

from model import FundamentalMatrix
from solver.solver_engine import SolverEngine


class SolverFundamentalMatrixPlaneParallax(SolverEngine):
    """ 平面视差法求解基础矩阵模型参数 """

    def __init__(self):
        self.homography = np.zeros([3, 3])
        self.is_homography_set = False

    def setHomography(self, homography):
        """ 设置单应变换矩阵 """
        self.is_homography_set = True
        self.homography = homography

    def returnMultipleModels(self):
        """ 确定是否有可能返回多个模型 """
        return True

    def sampleSize(self):
        """ 模型参数估计所需的最小样本数 """
        return 2

    def estimateModel(self,
                      points_,
                      sample_,
                      sample_number_,
                      weights_=None):
        """ 从给定的样本点，加权拟合模型参数

        参数
        ----------
        points : numpy
            输入的数据点集
        sample : list
            用于估计模型的样本点序号列表
        sample_number : int
            样本点的数目
        weights : list 可选
            数据点集中点的对应权重

        返回
        ----------
        list(Model)
            通过样本估计的模型列表
        """
        if not self.is_homography_set:
            print("单应变换矩阵未被定义\n")
            return []

        point_1 = points_[sample_[0]]
        point_2 = points_[sample_[1]]

        source_point_1 = np.array([point_1[0], point_1[1], 1])
        destination_point_1 = np.array([point_1[2], point_1[3], 1])
        source_point_2 = np.array([point_2[0], point_2[1], 1])
        destination_point_2 = np.array([point_2[2], point_2[3], 1])

        # 用单应矩阵投影点
        projected_point_1 = np.dot(self.homography, source_point_1)
        projected_point_2 = np.dot(self.homography, source_point_2)

        # 计算投影点与原点之间的直线
        line_1 = np.cross(projected_point_1, destination_point_1)
        line_2 = np.cross(projected_point_2, destination_point_2)

        # 估计外极
        epipole = np.cross(line_1, line_2)

        # 如果没有交集
        if abs(epipole[2]) < sys.float_info.epsilon:
            return []

        # 外极的叉积矩阵
        epipolar_cross = np.array([[0, -epipole[2], epipole[1]],
                                   [epipole[2], 0, -epipole[0]],
                                   [-epipole[1], epipole[0], 0]])

        # 计算 fundamental matrix
        fundamental_matrix = np.dot(epipolar_cross, self.homography)
        model = FundamentalMatrix(matrix=fundamental_matrix)
        return [model]
