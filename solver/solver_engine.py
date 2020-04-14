class SolverEngine:
    """ 模型参数求解器基类 """

    def __init__(self):
        pass
    
    def returnMultipleModels(self):
        """ 确定是否有可能返回多个模型 """
        pass

    def sampleSize(self):
        """ 模型参数估计所需的最小样本数 """
        return 0

    def estimateModel(self,
                      data,
                      sample,
                      sample_number,
                      models,
                      weights=None):
        """ 从给定的样本点，加权拟合模型参数 """
        pass