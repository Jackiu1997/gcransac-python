import math as m

import maxflow
import numpy as np


class GraphCut:
    """ 使用 maxflow 进行 graphcut 的能量方程最小化求解 """

    def __init__(self, node_max=0, edge_max=0):
        self.graph = maxflow.GraphFloat()
        self.Econst = 0

    
    def labeling(self,
                 points,
                 model,
                 estimator,
                 neighborhood_graph,
                 lambda_,
                 threshold):
        """ 通过maxflow对Graph进行能量最小化求解，标记模型内点
        
        参数
        ----------
        points : numpy
            输入的数据点集
        model : Model
            当前的模型参数
        estimator : Estimator
            用于估计该模型的模型估计器
        neighborhood_graph : GridNeighborGraph
            当前数据点集的领域图
        lambda_ : float
            空间相干性能量项的权重
        threshold : float
            决定内点和外点的阈值

        返回
        ----------
        list
            标记的模型内点序号列表
        """
        self.graph = maxflow.GraphFloat() # 先构建空 Graph
        squared_truncated_threshold = threshold ** 2
        point_number = np.shape(points)[0]
        self.graph.add_nodes(point_number)

        # 计算所有点的能量 Ep
        squared_residuals = []
        for point_idx in range(point_number):
            squared_distance = estimator.squaredResidual(points[point_idx], model)
            c0 = squared_distance / squared_truncated_threshold
            c1 = 1.0 - c0
            squared_residuals.append(squared_distance)
            self.__addTerm1(point_idx, c1, 0)
        # 空间相干性项权重须为正数，才能有效惩罚
        if lambda_ > 0:
            e00, e01, e10, e11 = 0.0, 1.0, 1.0, 0.0
            # 遍历所有点 p
            for point_idx in range(point_number):
                energy1 = max(0, 1.0 - squared_residuals[point_idx] / squared_truncated_threshold)

                # 遍历所有 p 点的边界邻居点，计算能量值
                neighbors = neighborhood_graph.getNeighbors(point_idx)
                for neighbor_idx in neighbors:
                    if neighbor_idx == point_idx:
                        continue
                    energy2 = 1.0 - squared_residuals[neighbor_idx] / squared_truncated_threshold
                    e00 = 0.5 * (energy1 + energy2)
                    e11 = 1.0 - e00
                    self.__addTerm2(point_idx, neighbor_idx, e00 * lambda_, lambda_, lambda_, e11 * lambda_)
        
        # 通过 maxflow 算法对图 G 进行能量最小化求解
        self.graph.maxflow()

        # 记录内点的序号
        inliers = []
        for point_idx in range(point_number):
            # 1 表示给定点接近 SINK
            if self.graph.get_segment(point_idx) == 0:
                inliers.append(point_idx)

        return inliers

    def __addTerm1(self, x, A, B):
        self.graph.add_tedge(x, A, B)

    def __addTerm2(self,
                 x, y,
                 A, B,
                 C, D):
        ''' E = A A  +  0   B-A
                D D     C-D 0
            Add edges for the first term
        '''
        self.graph.add_tedge(x, D, A)
        B -= A
        C -= D

        ''' now need to represent
            0 B
            C 0
        '''
        if (B < 0):
            ''' Write it as
                B B  +  -B 0  +  0   0
                0 0     -B 0     B+C 0
            '''
            self.graph.add_tedge(x, 0, B)       # first term
            self.graph.add_tedge(y, 0, -B)      # second term
            self.graph.add_edge(x, y, 0, B+C)   # third term
        elif (C < 0):
            ''' Write it as
                -C -C  +  C 0  +  0 B+C
                0  0     C 0     0 0
            '''
            self.graph.add_tedge(x, 0, -C)      # first term
            self.graph.add_tedge(y, 0, C)       # second term
            self.graph.add_edge(x, y, B+C, 0)   # third term
        else:
            ''' B >= 0, C >= 0 '''
            self.graph.add_edge(x, y, B, C)

    def __addTerm3(self,
                 x, y, z,
                 E000, E001,
                 E010, E011,
                 E100, E101,
                 E110, E111):
        pi = (E000 + E011 + E101 + E110) - (E100 + E010 + E001 + E111)
        delta = 0.0

        if (pi >= 0):
            self.Econst += E111 - (E011 + E101 + E110)

            self.graph.add_tedge(x, E101, E001)
            self.graph.add_tedge(y, E110, E100)
            self.graph.add_tedge(z, E011, E010)

            delta = (E010 + E001) - (E000 + E011) # -pi(E[x=0])
            self.graph.add_edge(y, z, delta, 0)
            delta = (E100 + E001) - (E000 + E101) # -pi(E[y=0])
            self.graph.add_edge(z, x, delta, 0)
            delta = (E100 + E010) - (E000 + E110) # -pi(E[z=0])
            self.graph.add_edge(x, y, delta, 0)

            if (pi > 0):
                # add new node and get node index
                u = self.graph.add_nodes(1).get_node_count() - 1
                self.graph.add_edge(x, u, pi, 0)
                self.graph.add_edge(y, u, pi, 0)
                self.graph.add_edge(z, u, pi, 0)
                self.graph.add_tedge(u, 0, pi)
        else:
            self.Econst += E000 - (E100 + E010 + E001)

            self.graph.add_tedge(x, E110, E010)
            self.graph.add_tedge(y, E011, E001)
            self.graph.add_tedge(z, E101, E100)

            delta = (E110 + E101) - (E100 + E111); # -pi(E[x=1])
            self.graph.add_edge(z, y, delta, 0)
            delta = (E110 + E011) - (E010 + E111); # -pi(E[y=1])
            self.graph.add_edge(x, z, delta, 0)
            delta = (E101 + E011) - (E001 + E111); # -pi(E[z=1])
            self.graph.add_edge(y, x, delta, 0)

            # add new node and get node index
            u = self.graph.add_nodes(1).get_node_count() - 1
            self.graph.add_edge(u, x, -pi, 0)
            self.graph.add_edge(u, y, -pi, 0)
            self.graph.add_edge(u, z, -pi, 0)
            self.graph.add_tedge(u, -pi, 0)
