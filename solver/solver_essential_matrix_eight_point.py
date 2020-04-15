import numpy as np
from numpy import linalg

from model import FundamentalMatrix
from solver.solver_engine import SolverEngine
import math as m


class SolverEssentialMatrixEightPoint(SolverEngine):
    """ 八点法求解本质矩阵模型参数 """

    def __init__(self):
        pass
    
    def returnMultipleModels(self):
        """ 确定是否有可能返回多个模型 """
        return False

    def sampleSize(self):
        """ 模型参数估计所需的最小样本数 """
        return 8

    def estimateModel(self,
                      points,
                      sample,
                      sample_number,
                      weights=None):
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

        ''' 1. 归一化 '''
        # 最小二乘模型拟合时，对点坐标进行归一化以实现数值稳定性
        normalized_sample_points, src_transform, dst_transform = self.__normalizeSamplePoints(
            points, sample)

        ''' 2. 求线性解 F' '''
        # 计算线性方程组参数矩阵 A
        coefficients = np.zeros([sample_number, 9])
        for i in range(sample_number):
            sample_idx = sample[i]
            weight = 1.0 if weights == None else weights[sample_idx]

            # 取点的坐标
            point = normalized_sample_points[i]
            x0 = point[0]
            y0 = point[1]
            x1 = point[2]
            y1 = point[3]

            # 设置参数行
            coefficients[i] = np.array(
                [x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, 1]) * weight

        # A * f' = 0，A 可能为奇异矩阵，所以采用 SVD 求解最小二乘解
        # f' 的解就是系数矩阵 A 最小奇异值对应的奇异向量，也就是 A 奇异值分解后 A=UDV^T 中矩阵 V 的最后一列矢量
        U, Sigma, VT = np.linalg.svd(coefficients)
        F = np.reshape(VT.T[:, -1], (3, 3))

        ''' 3. 强迫约束 '''
        U, Sigma, VT = np.linalg.svd(F)
        constrain = (Sigma[0]+Sigma[1]) / 2
        new_diag = np.diag(np.array([constrain, constrain, 0]))
        F = np.dot(np.dot(U, new_diag), VT)

        ''' 4. 解除归一化 '''
        F = np.dot(np.dot(dst_transform.T, F), src_transform)

        return [FundamentalMatrix(matrix=F)]

    def __normalizeSamplePoints(self, data, sample):
        ''' 归一化点集函数 '''
        sample_number = len(sample)

        # 初始化质点坐标
        mass_point_src = np.zeros(2)  # 第一张图片质点
        mass_point_dst = np.zeros(2)  # 第二张图片质点

        # 计算质点坐标 均值
        for i in range(sample_number):
            cur_point = data[sample[i]]
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
        normalized_points = np.zeros([sample_number, 4])
        for i in range(sample_number):
            cur_point = data[sample[i], :]
            np1 = (cur_point[0:2] - mass_point_src) * ratio_src
            np2 = (cur_point[2:4] - mass_point_dst) * ratio_dst
            normalized_points[i] = np.r_[np1, np2]

        # 创建归一化转换
        src_transform = np.array([[ratio_src, 0, -ratio_src * mass_point_src[0]],
                            [0, ratio_src, -ratio_src * mass_point_src[1]],
                            [0, 0, 1]])

        dst_transform = np.array([[ratio_dst, 0, -ratio_dst * mass_point_dst[0]],
                            [0, ratio_dst, -ratio_dst * mass_point_dst[1]],
                            [0, 0, 1]])
        # 返回归一化坐标，源图像转换矩阵，目标图像转换矩阵
        return normalized_points, src_transform, dst_transform
