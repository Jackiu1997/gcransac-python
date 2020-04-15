import numpy as np
from numpy import linalg
from sympy import Matrix

from model import EssentialMatrix

from .solver_engine import SolverEngine


class SolverEssentialMatrixFivePointStewenius(SolverEngine):
	""" 五点法求解基础矩阵模型参数 """

	def __init__(self):
		pass
	
	def returnMultipleModels(self):
		""" 确定是否有可能返回多个模型 """
		return True

	def sampleSize(self):
		""" 模型参数估计所需的最小样本数 """
		return 5
	
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
			sample = [i for i in range(sample_number)]
		if weights == None:
			weights = [1.0 for i in range(np.shape(points)[0])]
		coefficients = np.zeros([sample_number, 9])

		''' 创建包含极线约束的 nx9 矩阵 '''
		coefficients = np.zeros([sample_number, 9])
		for i in range(sample_number):
			sample_idx = sample[i]
			weight = 1.0 if weights == None else weights[sample_idx]

			# 取点的坐标
			point = points[i]
			x0 = point[0]
			y0 = point[1]
			x1 = point[2]
			y1 = point[3]

			# 设置参数行
			coefficients[i] = np.array(
				[x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, 1]) * weight

		# 从最小采样（使用LU）或非最小采样（使用SVD）中提取 nullspace
		nullSpace = np.zeros([9, 4])
		if sample_number == 5:
			res = Matrix(coefficients).nullspace()
			nullSpace = np.array(res[0].tolist())
			for i in range(1, 4):
				nullSpace = np.hstack((nullSpace, np.array(res[i].tolist())))
		else:
			U, Sigma, V = np.linalg.svd(np.dot(coefficients.T, coefficients), full_matrices=True)
			nullSpace = V[:, -4:]

		nullSpaceMatrix = [
			[nullSpace[0], nullSpace[3], nullSpace[6]],
			[nullSpace[1], nullSpace[4], nullSpace[7]],
			[nullSpace[2], nullSpace[5], nullSpace[8]],
		]
		
		# Step 2. 对行列式和迹的极线约束展开
		constraintMatrix = self.__buildConstraintMatrix(nullSpaceMatrix)

		# Step 3. 消除矩阵的一部分以分离z中的多项式
		eliminatedMatrix = np.zeros([10, 10])
		args = constraintMatrix[:10, :10]
		for i in range(10):
			b = np.reshape(constraintMatrix[:10, 10+i], [10])
			eliminatedMatrix[i] = linalg.solve(args, b)

		actionMatrix = np.zeros([10, 10])
		actionMatrix[:3, :10] = eliminatedMatrix[:3, :10]
		actionMatrix[3] = eliminatedMatrix[4]
		actionMatrix[4] = eliminatedMatrix[5]
		actionMatrix[5] = eliminatedMatrix[7]
		actionMatrix[6, 0] = -1.0
		actionMatrix[7, 1] = -1.0
		actionMatrix[8, 3] = -1.0
		actionMatrix[9, 6] = -1.0

		eigen_vals, eigen_vectors = linalg.eig(actionMatrix)

		# 现在我们有了x，y，z，我们需要将它们替换回空空间，以获得一个有效的基本矩阵解
		models = []
		for i in range(10):
			if eigen_vals[i].imag != 0:
				continue
			
			E_dst_src = np.dot(nullSpace, eigen_vectors[-4:, i].real)
			model = EssentialMatrix(matrix=np.reshape(E_dst_src, [3, 3]))
			models.append(model)
		return models
		
	def __multiplyDegOnePoly(self, a, b):
		''' Multiply two degree one polynomials of variables x, y, z.
			E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
			Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
		
			input: a - vector 4, b - vector 4
			output: vector 10
		'''
		output = np.zeros(10)
		output[0] = a[0] * b[0]                 # x^2
		output[1] = a[0] * b[1] + a[1] * b[0]   # xy
		output[2] = a[1] * b[1]                 # y^2
		output[3] = a[0] * b[2] + a[2] * b[0]   # xz
		output[4] = a[1] * b[2] + a[2] * b[1]   # yz
		output[5] = a[2] * b[2]                 # z^2
		output[6] = a[0] * b[3] + a[3] * b[0]   # x
		output[7] = a[1] * b[3] + a[3] * b[1]   # y
		output[8] = a[2] * b[3] + a[3] * b[2]   # z
		output[9] = a[3] * b[3]                 # 1
		return output

	def __multiplyDegTwoDegOnePoly(self, a,b):
		''' Multiply a 2 deg poly (in x, y, z) and a one deg poly in GrevLex order.
			Output order: x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1

			input: a - vector 10, b - vector 4
			output: vector 20
		'''
		output = np.zeros(20)
		output[0] = a[0] * b[0] # x^3
		output[1] = a[0] * b[1] + a[1] * b[0] # x^2y
		output[2] = a[1] * b[1] + a[2] * b[0] # xy^2
		output[3] = a[2] * b[1] # y^3
		output[4] = a[0] * b[2] + a[3] * b[0] # x^2z
		output[5] = a[1] * b[2] + a[3] * b[1] + a[4] * b[0] # xyz
		output[6] = a[2] * b[2] + a[4] * b[1] # y^2z
		output[7] = a[3] * b[2] + a[5] * b[0] # xz^2
		output[8] = a[4] * b[2] + a[5] * b[1] # yz^2
		output[9] = a[5] * b[2] # z^3
		output[10] = a[0] * b[3] + a[6] * b[0] # x^2
		output[11] = a[1] * b[3] + a[6] * b[1] + a[7] * b[0] # xy
		output[12] = a[2] * b[3] + a[7] * b[1] # y^2
		output[13] = a[3] * b[3] + a[6] * b[2] + a[8] * b[0] # xz
		output[14] = a[4] * b[3] + a[7] * b[2] + a[8] * b[1] # yz
		output[15] = a[5] * b[3] + a[8] * b[2] # z^2
		output[16] = a[6] * b[3] + a[9] * b[0] # x
		output[17] = a[7] * b[3] + a[9] * b[1] # y
		output[18] = a[8] * b[3] + a[9] * b[2] # z
		output[19] = a[9] * b[3] # 1
		return output

	def __computeEETranspose(self, nullSpace, i, j):
		# Shorthand for multiplying the Essential matrix with its transpose.
		# input: nullspace - matrix 1 4 * (3, 3)
		# output: matrix 1 10
		return self.__multiplyDegOnePoly(nullSpace[i][0], nullSpace[j][0]) +\
			self.__multiplyDegOnePoly(nullSpace[i][1], nullSpace[j][1]) +\
			self.__multiplyDegOnePoly(nullSpace[i][2], nullSpace[j][2])

	def __getTraceConstraint(self, nullSpace):
		# Builds the trace constraint: EEtE - 1/2 trace(EEt)E = 0
		# input: nullspace - matrix 1 4 * (3, 3)
		# output: matrix 9 20
		traceConstraint = np.zeros([9, 20])

		# Compute EEt.
		eet = [[0 for j in range(3)] for i in range(3)]
		for i in range(3):
			for j in range(3):
				eet[i][j] = 2 * self.__computeEETranspose(nullSpace, i, j)

		# Compute the trace.
		trace = eet[0][0] + eet[1][1] + eet[2][2]

		# Multiply EEt with E.
		for i in range(3):
			for j in range(3):
				traceConstraint[3*i+j] = self.__multiplyDegTwoDegOnePoly(eet[i][0], nullSpace[0][j]) +\
					self.__multiplyDegTwoDegOnePoly(eet[i][1], nullSpace[1][j]) +\
					self.__multiplyDegTwoDegOnePoly(eet[i][2], nullSpace[2][j]) -\
					0.5 * self.__multiplyDegTwoDegOnePoly(trace, nullSpace[i][j])

		return traceConstraint

	def __getDeterminantConstraint(self, nullSpace):
		# input: nullspace - matrix 1 4 * (3, 3)
		# output: matrix 1 20
		# Singularity constraint.
		return self.__multiplyDegTwoDegOnePoly(
				self.__multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][2]) -
				self.__multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][1]),
				nullSpace[2][0]) +\
			self.__multiplyDegTwoDegOnePoly(
				self.__multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][0]) -
				self.__multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][2]),
				nullSpace[2][1]) +\
			self.__multiplyDegTwoDegOnePoly(
				self.__multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][1]) -
				self.__multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][0]),
				nullSpace[2][2])

	def __buildConstraintMatrix(self, nullSpace):
		# input: nullspace - matrix 1 4 * (3, 3)
		# output: matrix 10 20
		row9 = self.__getTraceConstraint(nullSpace)
		row1 = self.__getDeterminantConstraint(nullSpace)
		row1 = np.reshape(row1, (1, 20))
		constraintMatrix = np.r_[row9, row1]
		return constraintMatrix
