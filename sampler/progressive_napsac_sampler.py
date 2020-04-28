import math as m

import numpy as np

from sampler.sampler import Sampler
from sampler.prosac_sampler import ProsacSampler
from utils.uniform_random_generator import UniformRandomGenerator
from neighbor.grid_neighborhood_graph import GridNeighborhoodGraph


class ProgressiveNapsacSampler(Sampler):


    def __init__(self, 
                container, 
                layer_data, 
                sample_size, 
                source_image_width, 
                source_image_height, 
                destination_image_width, 
                destination_image_height, 
                sampler_length=20):
        """ 初始化 P-NAPSAC 采样器

        参数
        ----------
        container : numpy
            采样的数据点集
        sample_size : int
            采样的样本数
        source_image_width: int
            源图像宽度
        source_image_height: int
            源图像高度
        destination_image_width: int
            目标图像宽度
        destination_image_height: int
            目标图像高度
        sampler_length : int 可选
            prosac 最大迭代次数
        """
        super().__init__(container)
        self.random_generator = UniformRandomGenerator()

        # src_img 和 dst_img 的长宽大小
        self.source_image_width = source_image_width           
        self.source_image_height = source_image_height          
        self.destination_image_width = destination_image_width  
        self.destination_image_height = destination_image_height
        
        self.point_number = np.shape(container)[0] # 数据点的数目

        self.grid_layers = []                      # 重叠 GridNeighborhoodGraph
        self.layer_data = layer_data               # 每层 Grid cell 的数目
        self.layer_number = len(layer_data)        # 重叠 GridNeighborhoodGraph 的层数

        self.current_layer_per_point = [0 for x in range(self.point_number)]         # 存储每个点的 layer 索引
        self.hits_per_point = [0 for x in range(self.point_number)]                  # 存储选择每个点的次数
        self.subset_size_per_point = [sample_size for x in range(self.point_number)] # 存储每个点的子集大小（邻域区域的大小）
        self.growth_function_progressive_napsac = []                                 # The P-NAPSAC growth function

        self.sample_size = sample_size             # 拟合模型的最小样本数目
        self.sampler_length = sampler_length       # 完全混合到全局采样的长度
        
        self.kth_sample_number = 0                 # PROSAC 抽样的迭代次数
        self.max_progressive_napsac_iterations = 0 # 应用全局采样之前本地采样迭代的最大次数

        # PROSAC 采样器，用于选择初始点，即超球面的中心
        self.one_point_prosac_sampler = ProsacSampler(
            container, 1, ransac_convergence_iterations=self.point_number)
        # 当采样完全混合为全局采样时使用的 PROSAC 采样器
        self.prosac_sampler = ProsacSampler(
            container, sample_size, ransac_convergence_iterations=self.point_number)

        self.initialized = self.__initialize(container)

    def __initialize(self, container):
        """ P-NAPSAC 初始化 layer_data 和 growth_function """
        # 初始化随机选择器，适应点集数
        self.random_generator.resetGenerator(0, self.point_number - 1)
        self.max_progressive_napsac_iterations = self.sampler_length * self.point_number

        # 构建 grid_layer
        for layer_idx in range(self.layer_number):
            cell_number_in_grid = self.layer_data[layer_idx]
            self.grid_layers.append(GridNeighborhoodGraph(container,
                                                          self.source_image_width / cell_number_in_grid,
                                                          self.source_image_height / cell_number_in_grid,
                                                          self.destination_image_width / cell_number_in_grid,
                                                          self.destination_image_height / cell_number_in_grid,
                                                          cell_number_in_grid))
        
        # 初始化 P-NAPSAC growth function
        self.growth_function_progressive_napsac = [0 for i in range(self.point_number)]
        local_sample_size = self.sample_size - 1
        T_n = float(self.max_progressive_napsac_iterations)
        for i in range(local_sample_size):
            T_n *= float(local_sample_size - i) / (self.point_number - i)
        
        T_n_prime = 1
        for i in range(self.point_number):
            if i + 1 <= local_sample_size:
                self.growth_function_progressive_napsac[i] = T_n_prime
                continue
            Tn_plus1 = float(i + 1) * T_n / (i + 1 - local_sample_size)
            self.growth_function_progressive_napsac[i] = T_n_prime + m.ceil(Tn_plus1 - T_n)
            T_n = Tn_plus1
            T_n_prime = self.growth_function_progressive_napsac[i]

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
        self.kth_sample_number += 1

        if sample_size != self.sample_size:
            print("采样失败，P-NAPSAC采样器未初始化\n")
            return []
        if sample_size > len(pool):
            print("采样失败，采样池数据小于所需样本\n")
            return []

        # 如果 P-NAPSAC 迭代次数到了，则采样 PROSAC 采样
        if self.kth_sample_number > self.max_progressive_napsac_iterations:
            self.prosac_sampler.setSampleNumber(self.kth_sample_number)
            subset = self.prosac_sampler.sample(pool, sample_size)
            return subset

        # Selection of the sphere center
        # 1: Let pi be a random point. ? e.g., selected by PROSAC.
        # 在超球上局部采样，仅选择一个点作为中心 
        center_point = self.one_point_prosac_sampler.sample(pool, 1)
        if len(center_point) == 0:
            return []

        # 2: ti := ti + 1
        # 获取选定的起始点
        initial_point = pool[center_point[0]]     # 选中点集合的中心
        self.hits_per_point[initial_point] += 1   # 增加选定点的命中次数
        hits = self.hits_per_point[initial_point]

        # Semi-random sample Mti of size m:
        # 获取 P-NAPSAC 的样本大小
        subset_size_progressive_napsac = self.subset_size_per_point[initial_point]
        while (hits > self.growth_function_progressive_napsac[subset_size_progressive_napsac - 1] and
            subset_size_progressive_napsac < self.point_number):
            subset_size_progressive_napsac = min(subset_size_progressive_napsac + 1, self.point_number)

        # 从 layerdata 中获取初始点的相邻点集
        current_layer = self.current_layer_per_point[initial_point]
        is_last_layer = False
        while True:
            # 如果选择为最后一个 layer，则使用 PROSAC 采样
            if current_layer >= self.layer_number:
                is_last_layer = True
                break

            # 获取初始点当前所在 layer 的相邻点集
            neighbors = self.grid_layers[current_layer].getNeighbors(initial_point)
            # 如果当前 layer 没有足够的相邻点，则使用更稀疏的 GridLayer
            if len(neighbors) < subset_size_progressive_napsac:
                current_layer += 1
                continue
            else:
                break

        # 如果未选择最后一层，则从初始点的相邻点中采样
        if not is_last_layer:
            neighbors = self.grid_layers[current_layer].getNeighbors(initial_point)
            # 随机选择 n-2 个点
            subset = self.random_generator.generateUniqueRandomSet(sample_size - 2,
                                                            max=subset_size_progressive_napsac - 2,
                                                            to_skip=initial_point)
            subset.append(initial_point)                                    # 初始点放在采样样本的末尾
            subset.append(neighbors[subset_size_progressive_napsac - 1])    # 离起始点最远的相邻点

            for i in range(sample_size - 2):
                subset[i] = neighbors[subset[i]]                # 用点的索引替换相邻索引
                self.hits_per_point[subset[i]] += 1             # 增加每个选择点的命中率
            self.hits_per_point[subset[sample_size - 2]] += 1   # 增加每个选择点的命中率
            return subset
        # 如果选择了最后一层，则使用 PROSAC 采样器进行全局采样
        else:
            self.prosac_sampler.setSampleNumber(self.kth_sample_number)
            subset = self.prosac_sampler.sample(pool, sample_size)
            subset[self.sample_size - 1] = initial_point
            return subset
