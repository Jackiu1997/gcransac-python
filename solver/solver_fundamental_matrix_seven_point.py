import math as m

import numpy as np
from numpy import linalg

from model import FundamentalMatrix
from solver.solver_engine import SolverEngine


class SolverFundamentalMatrixSevenPoint(SolverEngine):
	""" 七点法求解基础矩阵模型参数 """

	def __init__(self):
		pass
	
	def returnMultipleModels(self):
		""" 确定是否有可能返回多个模型 """
		return True

	def sampleSize(self):
		""" 模型参数估计所需的最小样本数 """
		return 7

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
		# TODO: fix model predict, and inliers preidct 0 bug
		if sample == None:
			sample = [i for i in range(sample_number)]
		if weights == None:
			weights = [1.0 for i in range(np.shape(points)[0])]
		coefficients = np.zeros([sample_number, 9])
		
		# 构成线性系统：a（=a）的第i行表示方程：（m2[i]，1）'*F*（m1[i]，1）=0
		for i in range(7):
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

		f1 = V[:, 6]
		f2 = V[:, 7]

		# f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary f. matrix.
		# as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
		# so f ~ lambda*f1 + (1 - lambda)*f2.
		# use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
		# it will be a cubic equation.
		# find c - polynomial coefficients.
		c = np.zeros(4)
		f1 -= f2

		t0 = f2[4] * f2[8] - f2[5] * f2[7]
		t1 = f2[3] * f2[8] - f2[5] * f2[6]
		t2 = f2[3] * f2[7] - f2[4] * f2[6]

		c[0] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2

		c[1] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2 -\
			f1[3] * (f2[1] * f2[8] - f2[2] * f2[7]) +\
			f1[4] * (f2[0] * f2[8] - f2[2] * f2[6]) -\
			f1[5] * (f2[0] * f2[7] - f2[1] * f2[6]) +\
			f1[6] * (f2[1] * f2[5] - f2[2] * f2[4]) -\
			f1[7] * (f2[0] * f2[5] - f2[2] * f2[3]) +\
			f1[8] * (f2[0] * f2[4] - f2[1] * f2[3])

		t0 = f1[4] * f1[8] - f1[5] * f1[7]
		t1 = f1[3] * f1[8] - f1[5] * f1[6]
		t2 = f1[3] * f1[7] - f1[4] * f1[6]

		c[2] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 -\
			f2[3] * (f1[1] * f1[8] - f1[2] * f1[7]) +\
			f2[4] * (f1[0] * f1[8] - f1[2] * f1[6]) -\
			f2[5] * (f1[0] * f1[7] - f1[1] * f1[6]) +\
			f2[6] * (f1[1] * f1[5] - f1[2] * f1[4]) -\
			f2[7] * (f1[0] * f1[5] - f1[2] * f1[3]) +\
			f2[8] * (f1[0] * f1[4] - f1[1] * f1[3])

		c[3] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2

		# 解三次方程；可以有1到3个根
		real_roots = np.roots(c)
		if len(real_roots) < 1 or len(real_roots) > 3:
			return None

		models = []
		f = np.ones(9)
		# 对每个实根求解基础矩阵
		for root in real_roots:
			lambda_ = root
			s = f1[8] * root + f2[8]
			mu = 1.0 / s
			# normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
			if m.fabs(s) > 1e-20:
				lambda_ *= mu
				for i in range(8):
					f[i] = f1[i] * lambda_ + f2[i] * mu
				models.append(FundamentalMatrix(matrix=np.reshape(f, (3,3))))

		return models
