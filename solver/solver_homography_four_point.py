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
		return False

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
		if sample == None:
			sample = [i for i in range(sample_number)]
		if weights == None:
			weights = [1.0 for i in range(np.shape(points)[0])]
		coefficients = np.zeros([2 * sample_number, 8])
		inhomogeneous = np.zeros(2 * sample_number)

		row_idx = 0
		for i in range(sample_number):
			sample_idx = sample[i]
			weight = weights[sample_idx]

			# 取点的坐标
			point = points[sample_idx]
			x1 = point[0]
			y1 = point[1]
			x2 = point[2]
			y2 = point[3]

			# 参数矩阵设置
			coefficients[row_idx] = np.array(
				[-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1]) * weight
			inhomogeneous[row_idx] = -weight * x2
			row_idx += 1

			coefficients[row_idx] = np.array(
				[0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1]) * weight
			inhomogeneous[row_idx] = -weight * y2
			row_idx += 1

		# 参数矩阵 coefficients 和 Y inhomogeneous
		# 利用 QR 分解求解 x
		Q, R = linalg.qr(coefficients)
		h = np.dot(linalg.pinv(R), np.dot(Q.T, inhomogeneous)).tolist()
		h.append(1.0)

		model = Homography(matrix=np.array(h).reshape((3,3)))
		return [model]
