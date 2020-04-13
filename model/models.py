import numpy as np

class Model:
    """ RANSAC算法求解模型基类 """

    def __init__(self):
        self.descriptor = None


class FundamentalMatrix(Model):
    """ 特征点匹配的基本矩阵模型 """

    def __init__(self, matrix=np.zeros([3, 3])):
        super().__init__()
        self.descriptor = matrix
    

class EssentialMatrix(Model):
    """ 特征点匹配的本质矩阵模型 """

    def __init__(self, matrix=np.zeros([3, 3])):
        super().__init__(matrix=matrix)
        self.descriptor = matrix


class Homography(Model):
    """ 特征点匹配的单应矩阵模型 """

    def __init__(self, matrix=np.zeros([3, 3])):
        super().__init__()
        self.descriptor = matrix
