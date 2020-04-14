import numpy as np


class _GridCell:

    def __init__(self,
                 idx_along_axis_x1,
                 idx_along_axis_y1,
                 idx_along_axis_x2,
                 idx_along_axis_y2,
                 cell_number_along_axes):
        """ 临域图的网格格子存储结构

        参数
        ----------
        idx_along_axis_x1 : int
            源数据 x 轴方向的长度
        idx_along_axis_x1 : int
            源数据 y 轴方向的长度
        idx_along_axis_x2 : int
            目标数据 x 轴方向的长度
        idx_along_axis_y2 : int
            目标数据 y 轴方向的长度
        cell_number_along_axes : list
            上面每个轴对应的网格数目 [8,8,8,8]
        """
        self.index = 0       # 求解点为邻居的唯一标识符
        self.points_idx = [] # cell 存储对应点集 index
        self.idx_along_axis_x1 = idx_along_axis_x1  # cell 位于 src 图中的位置
        self.idx_along_axis_y1 = idx_along_axis_y1
        self.idx_along_axis_x2 = idx_along_axis_x2  # cell 位于 dst 图中的位置
        self.idx_along_axis_y2 = idx_along_axis_y2

        # 构造函数：cell轴数目一致 或 cell轴数目独立
        cell_number_along_axis_x1 = cell_number_along_axes[0]
        cell_number_along_axis_y1 = cell_number_along_axes[1]
        cell_number_along_axis_x2 = cell_number_along_axes[2]
        cell_number_along_axis_y2 = cell_number_along_axes[3]
        
        self.index = self.idx_along_axis_x1 +\
            cell_number_along_axis_x1 * self.idx_along_axis_y1 +\
            cell_number_along_axis_x1 * cell_number_along_axis_y1 * self.idx_along_axis_x2 +\
            cell_number_along_axis_x1 * cell_number_along_axis_y1 * cell_number_along_axis_x2 * self.idx_along_axis_y2

    def __eq__(self, value):
        return (self.idx_along_axis_x1 == value.idx_along_axis_x1 and
                self.idx_along_axis_y1 == value.idx_along_axis_y1 and
                self.idx_along_axis_x2 == value.idx_along_axis_x2 and
                self.idx_along_axis_y2 == value.idx_along_axis_y2)

    def __lt(self, value):
        if (self.idx_along_axis_x1 < value.idx_along_axis_x1):
            return True
        if (self.idx_along_axis_x1 == value.idx_along_axis_x1 and
                self.idx_along_axis_y1 < value.idx_along_axis_y1):
            return True
        if (self.idx_along_axis_x1 == value.idx_along_axis_x1 and
            self.idx_along_axis_y1 == value.idx_along_axis_y1 and
                self.idx_along_axis_x2 < value.idx_along_axis_x2):
            return True
        if (self.idx_along_axis_x1 == value.idx_along_axis_x1 and
            self.idx_along_axis_y1 == value.idx_along_axis_y1 and
            self.idx_along_axis_x2 == value.idx_along_axis_x2 and
                self.idx_along_axis_y2 < value.idx_along_axis_y2):
            return True
        return False


class GridNeighborhoodGraph:
    """ 邻域图 """

    def __init__(self,
                 container,
                 cell_width_source_image,
                 cell_height_source_image,
                 cell_width_destination_image,
                 cell_height_destination_image,
                 cell_number_along_all_axes):
        self.cell_width_source_image = cell_width_source_image
        self.cell_height_source_image = cell_height_source_image
        self.cell_width_destination_image = cell_width_destination_image
        self.cell_height_destination_image = cell_height_destination_image
        self.cell_number_along_all_axes = [
            cell_number_along_all_axes for i in range(4)]
        self.container = container      # 存储的点集
        self.neighbor_number = 0        # 所有相邻的点边
        self.grid = []                  # 存储 GridCell 集合

        self.initialized = self.__initialize(self.container)

    def __initialize(self, container_):
        # 读取所有存储的点并存储入网格
        for row in range(np.shape(container_)[0]):
            col = 0
            idx_along_axis_x1 = container_[row, col] // self.cell_width_source_image
            idx_along_axis_y1 = container_[row, col+1] // self.cell_height_source_image
            idx_along_axis_x2 = container_[row, col+2] // self.cell_width_destination_image
            idx_along_axis_y2 = container_[row, col+3] // self.cell_height_destination_image

            # 构建图对应的网格结构
            cell = _GridCell(idx_along_axis_x1, idx_along_axis_y1,
                     idx_along_axis_x2, idx_along_axis_y2,
                     self.cell_number_along_all_axes)

            # 存储点序号到 Cell 中
            has_key = False
            for c in self.grid:
                if (cell.index == c.index):
                    c.points_idx.append(row)
                    has_key = True
            if not has_key:
                cell.points_idx.append(row)
                self.grid.append(cell)

            # 求解可能存在的边，即每个相邻的可能
            for cell in self.grid:
                n = len(cell.points_idx)
                self.neighbor_number += n * (n - 1) / 2
        return self.neighbor_number > 0

    def getNeighbors(self, point_idx):
        """ 获取临域图中此序号点的Cell的邻居点序号集合

        参数
        ----------
        point_idx : int
            查询的点序号

        返回
        ----------
        list
            查询点所有邻居点序号列表
        """
        for cell in self.grid:
            if point_idx in cell.points_idx:
                return cell.points_idx
        return []
