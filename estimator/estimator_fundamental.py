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
        models = self.minimal_solver.estimateModel(data, sample, sample_size)

        # 朝向约束检验 
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

        # 在应用最小二乘模型拟合时，对点坐标进行归一化以实现数值稳定性
        normalized_points, normalizing_transform_source, normalizing_transform_destination = self.__normalizePoints(
            data, sample, sample_number)
        if normalized_points == None:
            return []

        models = self.non_minimal_solver.estimateModel(normalized_points,
                                                       None,
                                                       sample_number,
                                                       weights=weights)
        # 估计基本矩阵的反归一化
        for model in models:
            model.descriptor = np.dot(np.linalg.inv(normalizing_transform_destination), model.descriptor)
            model.descriptor = np.dot(model.descriptor, normalizing_transform_source)
            model.descriptor = model.descriptor.normalize()
            if model.descriptor[2, 2] < 0:
                model.descriptor = -model.descriptor
        return models
    
    def residual(self, point, model):
        ''' 给定模型和数据点，计算误差 '''
        return m.sqrt(self.__squaredSampsonDistance(point, model.descriptor))

    def squaredResidual(self, point, model):
        """ 给定模型和数据点，计算误差的平方 """
        return self.__squaredSampsonDistance(point, model.descriptor)

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
        passed = False
        inlier_number = 0
        descriptor = model.descriptor
        # 当使用对称极距而不是 Sampson 距离时，也应该是内点的最小数
        minimum_inlier_number = max(self.sampleSize(), len(inliers) * self.minimum_inlier_ratio_in_validity_check)
        squared_threshold = threshold ** 2

        # 遍历由 sampson 距离确定的内点
        for idx in inliers:
            # 计算对称极距，并确定内点数目（如果内点数大于最小内点，则模型通过）
            if self.__squaredSymmetricEpipolarDistance(data[idx], descriptor) < squared_threshold:
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
    def __squaredSampsonDistance(self, point, descriptor):
        """ 点对应与本质矩阵的 sampson 距离平方 """
        x1 = point[0]
        y1 = point[1]
        x2 = point[2]
        y2 = point[3]

        e11 = descriptor[0, 0]
        e12 = descriptor[0, 1]
        e13 = descriptor[0, 2]
        e21 = descriptor[1, 0]
        e22 = descriptor[1, 1]
        e23 = descriptor[1, 2]
        e31 = descriptor[2, 0]
        e32 = descriptor[2, 1]
        e33 = descriptor[2, 2]

        rxc = e11 * x2 + e21 * y2 + e31
        ryc = e12 * x2 + e22 * y2 + e32
        rwc = e13 * x2 + e23 * y2 + e33
        r = (x1 * rxc + y1 * ryc + rwc)
        rx = e11 * x1 + e12 * y1 + e13
        ry = e21 * x1 + e22 * y1 + e23

        return r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry)

    def __squaredSymmetricEpipolarDistance(self, point, descriptor):
        """ 点对应与本质矩阵的对称极距平方 """
        x1 = point[0]
        y1 = point[1]
        x2 = point[2]
        y2 = point[3]

        e11 = descriptor[0, 0]
        e12 = descriptor[0, 1]
        e13 = descriptor[0, 2]
        e21 = descriptor[1, 0]
        e22 = descriptor[1, 1]
        e23 = descriptor[1, 2]
        e31 = descriptor[2, 0]
        e32 = descriptor[2, 1]
        e33 = descriptor[2, 2]

        rxc = e11 * x2 + e21 * y2 + e31
        ryc = e12 * x2 + e22 * y2 + e32
        rwc = e13 * x2 + e23 * y2 + e33
        r = (x1 * rxc + y1 * ryc + rwc)
        rx = e11 * x1 + e12 * y1 + e13
        ry = e21 * x1 + e22 * y1 + e23
        a = rxc * rxc + ryc * ryc
        b = rx * rx + ry * ry

        return r * r * (a + b) / (a * b)
    
    ''' 规范化点集函数 '''
    def __normalizePoints(self, data, sample, sample_number):
        ''' 规范化点集函数 '''
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
        normalized_points_ = np.zeros([sample_number, 4])
        for i in range(sample_number):
            cur_point = data[sample[i], :]
            np1 = (cur_point[0:2] - mass_point_src) * ratio_src
            np2 = (cur_point[2:4] - mass_point_dst) * ratio_dst
            normalized_points_[i] = np.r_[np1, np2]

        # 创建归一化转换
        normalizing_transform_source_ = np.array([[ratio_src, 0, -ratio_src * mass_point_src[0]],
                                                  [0, ratio_src, -ratio_src * mass_point_src[1]],
                                                  [0, 0, 1]])

        normalizing_transform_destination_ = np.array([[ratio_dst, 0, -ratio_dst * mass_point_dst[0]],
                                                       [0, ratio_dst, -ratio_dst * mass_point_dst[1]],
                                                       [0, 0, 1]])
        # 返回归一化坐标，源图像转换矩阵，目标图像转换矩阵
        return normalized_points_, normalizing_transform_source_, normalizing_transform_destination_

    ''' 定向极线约束函数 Oriented epipolar constraints '''
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
