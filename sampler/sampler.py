class Sampler:
    """ 采样器基类 """
    
    def __init__(self, containter_):
        self.container = containter_ # 采样的数据集
        self.initialized = False     # 采样器是否被初始化

    def __initialize(self, containter_):
        """ 初始化样本构建，必须在样本被调用前"""
        pass

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
        pass
