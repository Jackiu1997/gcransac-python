import numpy as np
from numpy import linalg

from model import *
from solver.solver_engine import SolverEngine


class SolverHomographyFourPoint(SolverEngine):
    """ 四点法求解单应矩阵模型参数 """

    def __init__(self):
        pass
    
    def returnMultipleModels(self):
        """ 确定是否有可能返回多个模型 """
        pass

    def sampleSize(self):
        """ 模型参数估计所需的最小样本数 """
        return 4

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
        weights : list
            数据点集中点的对应权重

        返回
        ----------
        list(Model)
            通过样本估计的模型列表
        """
        if sample != None:
            sample_number = len(sample)
        row_number = 2 * sample_number
        coefficients = np.zeros([row_number, 8])
        inhomogeneous = np.zeros(row_number)

        row_idx = 0
        weight = 1.0
        for i in range(sample_number):
            idx = i if (sample == None) else sample[i]

            # 取点的坐标
            point = points[idx, :]
            x1 = point[0]
            y1 = point[1]
            x2 = point[2]
            y2 = point[3]

            # 权重初始化
            if weights != None:
                weight = weights[idx]
            minus_weight_times_x1 = -weight * x1
            minus_weight_times_y1 = -weight * y1
            weight_times_x2 = weight * x2
            weight_times_y2 = weight * y2

            # 参数矩阵设置
            coefficients[row_idx, 0] = minus_weight_times_x1
            coefficients[row_idx, 1] = minus_weight_times_y1
            coefficients[row_idx, 2] = -weight
            coefficients[row_idx, 3] = 0
            coefficients[row_idx, 4] = 0
            coefficients[row_idx, 5] = 0
            coefficients[row_idx, 6] = weight_times_x2 * x1
            coefficients[row_idx, 7] = weight_times_x2 * y1
            inhomogeneous[row_idx] = -weight_times_x2
            row_idx += 1

            coefficients[row_idx, 0] = 0
            coefficients[row_idx, 1] = 0
            coefficients[row_idx, 2] = 0
            coefficients[row_idx, 3] = minus_weight_times_x1
            coefficients[row_idx, 4] = minus_weight_times_y1
            coefficients[row_idx, 5] = -weight
            coefficients[row_idx, 6] = weight_times_y2 * x1
            coefficients[row_idx, 7] = weight_times_y2 * y1
            inhomogeneous[row_idx] = -weight_times_y2
            row_idx += 1

        # 参数矩阵 coefficients 和 Y inhomogeneous
        # 利用 QR 分解求解 x
        Q, R = linalg.qr(coefficients)
        h = np.dot(linalg.pinv(R), np.dot(Q.T, inhomogeneous)).tolist()
        h.append(1.0)

        model = Homography(matrix=np.array(h).reshape((3,3)))
        return [model]