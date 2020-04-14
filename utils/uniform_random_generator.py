import random


class UniformRandomGenerator:
    """ 均匀随机数产生器 """
    
    def __init__(self):
        self.range_min = 0       # 可取最小值
        self.range_max = 100000  # 可取最大值
        
    def resetGenerator(self, min, max):
        """ 设置随机数发生器的随机数范围

        参数
        ----------
        min : int
            可取最小值
        max : int
            可取最大值
        """
        self.range_min, self.range_max = min, max
        
    def generateUniqueRandomSet(self, sample_size, max = None, to_skip = -1):
        """ 产生一个均匀随机的随机数序列
        
        参数
        ----------
        sample_size : int
            选取样本大小
        max : int 可选
            可取最大值
        to_skip : int 可选
            不可选取的随机数

        返回
        ----------
        list
            产生的随机序列样本列表
        """
        # 如果输入了最大值，则重设随机数发生器范围
        if max != None:
            self.resetGenerator(0, max)
        i = 0
        sample = []
        while i < sample_size:
            rand_num = int(self.__getRandomNumber())
            # 如果产生的数不和前面重复，且不是需要跳过的数，则加入样本
            if rand_num in sample[0:i] or rand_num == to_skip:
                continue
            sample.append(rand_num)
            i += 1
        return sample

    def __getRandomNumber(self):
        """ 产生一个均匀随机数 """
        return random.uniform(self.range_min, self.range_max+1)
