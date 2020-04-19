
class Score:
    """ RANSAC Scoring """

    def __init__(self):
        self.inlier_number = 0   # 内点数目
        self.value = 0.0         # 得分

    def __lt__(self, v):
        return self.value < v.value and\
            self.inlier_number < v.inlier_number

    def __gt__(self, v):
        return self.value > v.value

    def __eq__(self, v):
        return self.value == v.value


class MSACScoringFunction:

    def __init__(self):
        self.squared_truncated_threshold = 0.0
        self.point_number = 0
		# 在计算分数时，只验证第k点。如果有一个时间敏感的应用程序，
        # 并且在点的子集上验证模型就足够了，那么这可能是有益的。
        self.verify_every_kth_point = 1

    def setSkippingParameter(self, verify_every_kth_point):
	    self.verify_every_kth_point = verify_every_kth_point

    def initialize(self, squared_truncated_threshold, point_number):
        self.squared_truncated_threshold = squared_truncated_threshold
        self.point_number = point_number

    def getScore(self,
                 points,
                 model,
                 estimator,
                 threshold,
                 best_score):
        """ 求解模型对应的评估得分
        
        参数
        ----------
        points : numpy
            输入的数据点集
        model : Model
            当前模型参数
        estimator : Estimator
            模型的估计器
        threshold : float
            决定内点和外点的阈值
        best_score : Score
            目前的最佳模型得分

        返回
        ----------
        Score, list
            当前模型参数的评估得分
            当前模型参数的对应内点
        """
        score = Score()         # 当前得分
        inliers = []            # 选择的内点集合
        
        # 遍历所有点，计算残差平方
        for point_idx in range(0, self.point_number, self.verify_every_kth_point):
            # 计算点对模型的残差
            squared_residual = estimator.squaredResidual(points[point_idx], model)
            
            # 如果残差小于阈值，则将其存储为内点，并增加得分
            if squared_residual < self.squared_truncated_threshold:
                inliers.append(point_idx)
                score.inlier_number += 1
                # 加分: 原始截断二次损失如下：1 - 残差^2/阈值^2
                score.value += 1.0 - squared_residual / self.squared_truncated_threshold

            # 无更好的模型则终止计算过程
            if self.point_number - point_idx + score.inlier_number < best_score.inlier_number:
                return score, inliers

        return score, inliers
