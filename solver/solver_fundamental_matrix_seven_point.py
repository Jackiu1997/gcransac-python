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
		
		# A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
		# the solution is linear subspace of dimensionality 2.
		# => use the last two singular std::vectors as a basis of the space
		U, Sigma, VT = np.linalg.svd(coefficients)
		f1 = VT.T[:, 6]
		f2 = VT.T[:, 7]

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

		''' 4. 解除归一化 '''
		for model in models:
			model.descriptor = np.dot(np.dot(dst_transform.T, model.descriptor), src_transform)

		return models

	def __normalizeSamplePoints(self, data, sample):
		''' 归一化点集函数 '''
		sample_number = len(sample)

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
