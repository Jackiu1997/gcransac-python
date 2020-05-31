import math as m

import numpy as np

from .sampler import Sampler
from utils.uniform_random_generator import UniformRandomGenerator


class ProsacSampler(Sampler):
    """ PROSAC 渐进采样器 """

    def __init__(self, container, sample_size, ransac_convergence_iterations=100000):
        """ 初始化 PORSAC 采样器

        参数
        ----------
        container : numpy
            采样的数据点集
        sample_size : int
            采样的样本数
        ransac_convergence_iterations : int 可选
            prosac 最大迭代次数
        """
        super().__init__(container)
        self.random_generator = UniformRandomGenerator()

        self.sample_size = sample_size
        self.point_number = np.shape(container)[0]
        self.ransac_convergence_iterations = ransac_convergence_iterations
        self.kth_sample_number = 1      # prosac 采样迭代次数
        self.subset_size = 0            # 采样样本大小
        self.largest_sample_size = 0    # 最大样本数目
        self.growth_function = []       # PROSAC 增长函数

        self.initialized = self.initialize(container)

    def initialize(self, container):
        """ PROSAC 采样初始化 growth_function """
        self.growth_function = [0 for i in range(self.point_number)]

        # Tq.he data points in U_N are sorted in descending order w.r.t. the quality function 
        # Let {Mi}i = 1...T_N denote the sequence of samples Mi c U_N that are uniformly drawn by Ransac.

        # Let T_n be an average number of samples from {Mi}i=1...T_N that contain data points from U_n only.
        # compute initial value for T_n
        #                                  n - i
        # T_n = T_N * Product i = 0...m-1 -------, n >= sample size, N = points size
        #                                  N - i
        T_n = self.ransac_convergence_iterations
        for i in range(self.sample_size):
            T_n *= (self.sample_size - i) / (self.point_number - i)

        T_n_prime = 1
        # compute values using recurrent relation
        #             n + 1
        # T(n+1) = --------- T(n), m is sample size.
        #           n + 1 - m

        # growth function is defined as
        # g(t) = min {n, T'_(n) >= t}
        # T'_(n+1) = T'_(n) + (T_(n+1) - T_(n))
        for i in range(self.point_number):
            if i + 1 <= self.sample_size:
                self.growth_function[i] = T_n_prime
                continue
            Tn_plus1 = float(i + 1) * T_n / (i + 1 - self.sample_size)
            self.growth_function[i] = T_n_prime + m.ceil(Tn_plus1 - T_n)
            T_n = Tn_plus1
            T_n_prime = self.growth_function[i]

        self.largest_sample_size = self.sample_size  # PROSAC 中最大样本集合点数
        #self.subset_size = np.shape(container)[0]     # 当前采样池的点集大小
        self.subset_size = self.sample_size     # 当前采样池的点集大小		

        # 初始化随机数产生器
        self.random_generator.resetGenerator(0, self.subset_size - 1)
        return True

    def sample(self, pool, sample_size):
        """ 根据给定的采样池和样本大小进行采样

        参数
        ----------
        pool : list(int)
            采样的数据集合的序号池
        sample_size : int
            采样的样本数

        返回
        ----------
        list
            采样的数据集合序号列表
        """
        if sample_size != self.sample_size:
            print("采样错误，PROSAC采样器未被初始化")
            self.__incrementIterationNumber()
            return []
        
        # 如果 PROSAC 采样与 RANSAC，则采样均匀随机采样
        if self.kth_sample_number > self.ransac_convergence_iterations:
            subset = self.random_generator.generateUniqueRandomSet(sample_size)
            return subset
        else:
            # 产生 PROSAC 样本 [0, subset_size-2]
            subset = self.random_generator.generateUniqueRandomSet(self.sample_size - 1)
            # 最后一个索引是当前使用的子集末尾的点的索引
            subset.append(self.subset_size - 1)
            self.__incrementIterationNumber()
            return subset
            
    def setSampleNumber(self, k):
        """ 外部设置目前采样次数为第 k 次 PROSAC 采样"""
        self.kth_sample_number = k # 设置为第 k 次 PROSAC 采样

        # 如果与 RANSAC 完全相同，则设置随机生成器以从所有可能的索引生成值
        if self.kth_sample_number > self.ransac_convergence_iterations:
            self.random_generator.resetGenerator(0, self.point_number - 1)
        # 根据需要增加采样池的大小
        else:
            while self.kth_sample_number > self.growth_function[self.subset_size - 1]:
                self.subset_size += 1 # n = n + 1
                if self.subset_size > self.point_number:
                    self.subset_size = self.point_number
                if self.largest_sample_size < self.subset_size:
                    self.largest_sample_size = self.subset_size
                # 重置随机生成器以从当前点子集生成值，但最后一个除外，因为它将始终被使用
                self.random_generator.resetGenerator(0, self.subset_size - 2)

    def __incrementIterationNumber(self):
        self.kth_sample_number += 1 # PROSAC 迭代数自增

        # 如果与 RANSAC 完全相同，则设置随机生成器以从所有可能的索引生成值
        if self.kth_sample_number > self.ransac_convergence_iterations:
            self.random_generator.resetGenerator(0, self.point_number - 1)
        # 根据需要增加采样池的大小
        elif self.kth_sample_number > self.growth_function[self.subset_size - 1]:
            self.subset_size += 1 # n = n + 1
            if self.subset_size > self.point_number:
                self.subset_size = self.point_number
            if self.largest_sample_size < self.subset_size:
                self.largest_sample_size = self.subset_size
            # 重置随机生成器以从当前点子集生成值，但最后一个除外，因为它将始终被使用
            self.random_generator.resetGenerator(0, self.subset_size - 2)

   
