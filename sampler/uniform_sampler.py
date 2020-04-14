import numpy as np

from .sampler import Sampler
from utils.uniform_random_generator import UniformRandomGenerator


class UniformSampler(Sampler):
    """ 均匀随机采样器 """
    
    def __init__(self, container_):
        super().__init__(container_)
        self.random_generator = UniformRandomGenerator()
        self.initialized = self.__initialize(container_)
    
    def __initialize(self, container_):
        """ 初始化样本构建，必须在样本被调用前"""
        self.random_generator.resetGenerator(0, np.shape(self.container)[0] - 1)
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
        if sample_size > len(pool):
            return []
        # 生成点集序号的随机序列
        subset = self.random_generator.generateUniqueRandomSet(sample_size, max=len(pool)-1)
        # 用 pool 中的索引替换 subset 索引
        for i in range(sample_size):
            subset[i] = pool[subset[i]]
        return subset
