class Estimator:
    """ 模型估计器基类 """

    def __init__(self):
        pass

    def isWeightingApplicable(self):
        """ 当应用非最小拟合时决定点是否可以加权的标志 """
        pass

    def inlierLimit(self):
        """ 对非最小样本进行内部RANSAC时的样本大小 """
        pass

    def sampleSize(self):
        """ 估计模型所需的最小样本的大小 """
        pass

    def nonMinimalSampleSize(self):
        """ 估计模型所需的非最小样本的大小 """
        pass

    def estimateModel(self,
                      data,
                      sample):
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
        pass

    def estimateModelNonminimal(self,
                                data,
                                sample,
                                sample_number,
                                weights=None):
        """ 根据数据点集的非最小采样估计模型
            对于一条直线，在一组点上使用SVD而不是从两点构造一条直线。
            默认情况下，这只是实现最小情况。在加权最小二乘的情况下，权重可以输入到函数中

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
        pass

    def residual(self, data, model):
        """ 给定模型和数据点，计算误差 """
        pass

    def squaredResidual(self, data, model):
        """ 给定模型和数据点，计算误差的平方 """
        pass

    def isValidSample(self, data, sample):
        """ 在计算模型参数之前判断所选样本是否退化

        参数
        ----------
        data : numpy
            输入的数据点集
        sample : list
            用于估计模型的样本点序号列表

        返回
        ----------
        bool
            样本是否有效
        """
        return True

    def isValidModel(self,
                     model,
                     data=None,
                     inliers=None,
                     minimal_sample=None,
                     threshold=None):
        """ 检查模型是否有效，可以是模型结构的几何检查或其他验证

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
        return True
