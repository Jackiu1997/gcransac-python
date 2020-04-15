import math as m

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
import sys


class EstimatorFundamental(Estimator):
    """ 基础矩阵估计器 """

    def __init__(self,
                 minimalSolver,
                 nonMinimalSolver,
                 minimum_inlier_ratio_in_validity_check=0.5,
                 use_degensac=True,
                 homography_threshold=2.0):
        super().__init__()
        # 用于估计最小样本模型的估计器
        self.minimal_solver = minimalSolver()
        # 用于估计非最小样本模型的估计器
        self.non_minimal_solver = minimalSolver()
        # 通过有效性测试所需的内点比率的下限
        self.minimum_inlier_ratio_in_validity_check = minimum_inlier_ratio_in_validity_check
        # 是否使用 DEGENSAC 的标志，DEGENSAC处理模型的点来自单个平面或几乎来自单个平面的情况
        self.use_degensac = use_degensac
        # DEGENSAC 中决定一个采样是否 H-degenerate 的阈值
        self.homography_threshold = homography_threshold
        self.squared_homography_threshold = homography_threshold ** 2

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
    # TODO: fix model check all wrong bug
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
        # 当使用对称极距而不是 Sampson 距离时，也应该是内点的最小数
        minimum_inlier_number = max(self.sampleSize(), len(inliers) * self.minimum_inlier_ratio_in_validity_check)
        inlier_number = 0
        descriptor = model.descriptor
        squared_threshold = threshold ** 2
        passed = False

        # 遍历由 sampson 距离确定的内点
        for idx in inliers:
            # 计算对称极距，并确定内点数目（如果内点数大于最小内点，则模型通过）
            if self.__symmetricEpipolarDistance(data[idx], descriptor) < squared_threshold:
                inlier_number += 1
                if inlier_number >= minimum_inlier_number:
                    passed = True
                    break
        if not passed:
            return False

        # 通过检查模型是否接近单个平面来验证模型
        if self.use_degensac:
            return self.__applyDegensac(model,
                                        data,
                                        inliers,
                                        minimal_sample,
                                        threshold)

        return True

    def __applyDegensac(self, model, data, inliers, minimal_sample, threshold):
        """ 评估 H-degenerate 样本试验，必要时使用DEGENSAC

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
            模型是否接近单个平面
        """
        # 可能的三元组点
        triplets = [[0, 1, 2],
                    [3, 4, 5],
                    [0, 1, 6],
                    [3, 4, 6],
                    [2, 5, 6]]
        number_of_triplets = 5                  # 待测三元组数目
        fundamental_matrix = model.descriptor   # 最小样本的基本矩阵

        # 对估计基本矩阵进行奇异值分解
        U, Sigma, V = np.linalg.svd(model.descriptor, full_matrices=True)
        # 计算目标图像中的外极
        epipole = U[0:3, -1] / U
        
        # 极柱交叉生成矩阵的计算
        epipolar_cross = np.array([[0, -epipole[2], epipole[1]],
                                   [epipole[2], 0, -epipole[0]],
                                   [-epipole[1], epipole[0], 0]])
        A = np.dot(epipolar_cross, model.descriptor)

        # 决定样本是否 H-degenerate 的标志
        h_degenerate_sample = False
        best_homography_model = Homography()

        # 遍历样本中的点的三元组
        for triplet_idx in range(number_of_triplets):
            point_1_idx = minimal_sample[triplets[triplet_idx][0]]
            point_2_idx = minimal_sample[triplets[triplet_idx][1]]
            point_3_idx = minimal_sample[triplets[triplet_idx][2]]

            point_1 = data[point_1_idx]
            point_2 = data[point_2_idx]
            point_3 = data[point_3_idx]

            point_1_1 = np.array([point_1[0], point_1[1], 1])
            point_2_1 = np.array([point_1[2], point_1[3], 1])
            point_1_2 = np.array([point_2[0], point_2[1], 1])
            point_2_2 = np.array([point_2[2], point_2[3], 1])
            point_1_3 = np.array([point_3[0], point_3[1], 1])
            point_2_3 = np.array([point_3[2], point_3[3], 1])

            # 计算每个点的外极端点的叉积
            point_1_cross_epipole = np.cross(point_2_1, epipole)
            point_2_cross_epipole = np.cross(point_2_2, epipole)
            point_3_cross_epipole = np.cross(point_2_3, epipole)

            b = np.array([np.dot(np.cross(point_2_1, np.dot(A, point_1_1)).T, point_1_cross_epipole) / linalg.norm(point_1_cross_epipole),
                          np.dot(np.cross(point_2_2, np.dot(A, point_1_2)).T, point_2_cross_epipole) / linalg.norm(point_2_cross_epipole),
                          np.dot(np.cross(point_2_3, np.dot(A, point_1_3)).T, point_3_cross_epipole) / linalg.norm(point_3_cross_epipole)])

            M = np.vstack((point_1_1, point_1_2, point_1_3))
            
            homography = A - np.dot(epipole, (np.dot(linalg.inv(M), b)).T)

            # 与隐含单应矩阵一致的内点的个数
            inlier_number = 3
            for i in range(self.sampleSize()):
                # 获取最小样本的点序号
                idx = minimal_sample[i]

                # 检查该点是否不包括在当前三元组中，若是则不必计算误差
                if idx == point_1_idx or idx == point_2_idx or idx == point_3_idx:
                    continue
            
                # 计算重投影误差
                point = data[idx]
                x1 = point[0]
                y1 = point[1]
                x2 = point[2]
                y2 = point[3]

                # Calculating H * p
                t1 = homography(0, 0) * x1 + homography(0, 1) * y1 + homography(0, 2)
                t2 = homography(1, 0) * x1 + homography(1, 1) * y1 + homography(1, 2)
                t3 = homography(2, 0) * x1 + homography(2, 1) * y1 + homography(2, 2)
                # 计算投影点和原点的差
                d1 = x2 - (t1 / t3)
                d2 = y2 - (t2 / t3)
                # 计算重投影误差的平方
                squared_residual = d1 ** 2 + d2 ** 2

                # 如果二次投影的平方误差小于阈值，记点为内点
                if squared_residual < squared_homography_threshold:
                    inlier_number += 1

            # 如果至少有5个点对应关系与单应变换一致，则样本 H-degenerate
            if inlier_number >= 5:
                best_homography_model.descriptor = homography
                h_degenerate_sample = True
                break

        if h_degenerate_sample:
            # 定义单应矩阵估计器 用于计算残差，估计单应模型
            homography_estimator = EstimatorHomography(SolverHomographyFourPoint,
                                                       SolverHomographyFourPoint)

            # 迭代基本矩阵的内点，并选择那些是单应矩阵的内点
            homography_inliers = []
            for inlier_idx in inliers:
                if homography_estimator.squaredResidual(data[inlier_idx], best_homography_model) < squared_homography_threshold:
                    homography_inliers.append(inlier_idx)

            # 如果单应矩阵没有足够的内点来估计，则终止
            if len(homography_inliers) < homography_estimator.nonMinimalSampleSize():
                return False

            # 从提供的内点估计单应矩阵，非最小样本拟合的单应矩阵作为参考
            homographies = homography_estimator.estimateModelNonminimal(data,
                                                                        homography_inliers,
                                                                        len(homography_inliers))
            if len(homographies) == 0:
                return False

            # 局部GC-RANSAC，通过平面和视差算法使用确定的单应矩阵
            sampler = UniformSampler(data)

            estimator = EstimatorFundamental(SolverFundamentalMatrixPlaneParallax,
                                             SolverFundamentalMatrixEightPoint,
                                             minimum_inlier_ratio_in_validity_check=0.0,
                                             use_degensac=False)
            estimator.minimal_solver.setHomography(homographies[0].descriptor)

            gcransac = GCRANSAC()
            gcransac.settings.threshold = threshold
            gcransac.settings.spatial_coherence_weight = 0
            gcransac.settings.confidence = 0.99
            gcransac.settings.neighborhood_sphere_radius = 8

            lo_gc_model, lo_gc_inliers = gcransac.run(data,
                                                      estimator,
                                                      sampler,
                                                      sampler,
                                                      None)

            # 如果由更多的内点，则更新模型参数
            if len(lo_gc_inliers) > len(inliers):
                model = lo_gc_model

        return True, model

    def __enforceRankTwoConstraint(self, model):
        # 对基本矩阵进行奇异值分解
        U, Sigma, V = np.linalg.svd(model.descriptor, full_matrices=True)
        # 使最后一个奇异值为零
        diagonal = np.diag(Sigma)
        diagonal[2, 2] = 0.0
        # 从SVD分解中求出基本矩阵，使用新的奇异值
        model.descriptor = U * diagonal * V.T
        
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

        return x2_f_x1 ** 2 * (1 / (f_x1[0] ** 2 + f_x1[1] ** 2) + 1 / (x2_f[0] ** 2 + x2_f[1] ** 2))
    
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
