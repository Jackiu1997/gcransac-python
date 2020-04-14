import numpy as np
from numpy import linalg

from model import FundamentalMatrix
from solver.solver_engine import SolverEngine


class SolverFundamentalMatrixEightPoint(SolverEngine):
	""" 八点法求解基础矩阵模型参数 """

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
		if sample == None:
			sample_number = np.shape(points)[0]
			sample = [i for i in range(sample_number)]
		if weights == None:
			weights = [1.0 for i in range(np.shape(points)[0])]
		coefficients = np.zeros([sample_number, 9])
		
		# 构成线性系统：a（=a）的第i行表示方程：（m2[i]，1）'*F*（m1[i]，1）=0
		for i in range(sample_number):
			sample_idx = sample[i]
			weight = weights[sample_idx]

			# 取点的坐标
			point = points[sample_idx]
			x0 = point[0]
			y0 = point[1]
			x1 = point[2]
			y1 = point[3]

			# 设置参数矩阵
			weight_times_x0 = weight * x0
			weight_times_y0 = weight * y0
			weight_times_x1 = weight * x1
			weight_times_y1 = weight * y1

			coefficients[i, 0] = weight_times_x1 * x0
			coefficients[i, 1] = weight_times_x1 * y0
			coefficients[i, 2] = weight_times_x1
			coefficients[i, 3] = weight_times_y1 * x0
			coefficients[i, 4] = weight_times_y1 * y0
			coefficients[i, 5] = weight_times_y1
			coefficients[i, 6] = weight_times_x0
			coefficients[i, 7] = weight_times_y0
			coefficients[i, 8] = weight
		
		# A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
		# the solution is linear subspace of dimensionality 2.
		# => use the last two singular std::vectors as a basis of the space
		# (according to SVD properties)
		U, Sigma, V = np.linalg.svd(np.dot(coefficients.T, coefficients), full_matrices=True)
		nullSpace = V[:, -1]

		model = FundamentalMatrix(matrix=np.reshape(nullSpace, (3, 3)))
		return [model]
			