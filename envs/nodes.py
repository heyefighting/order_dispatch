"""
owner: Zou ying Cao
data: 2023-02-01
description: 路网划分
"""
from env_utils.neighbours import *


class Node(object):
    def __init__(self, index):
        self._index = index
        # 订单
        self.orders = []
        self.order_num = 0
        # 骑手
        self.couriers = {}
        self.online_courier_num = 0
        self.offline_courier_num = 0
        # 商家
        self.shops = {}
        self.shop_num = 0
        # 用户
        self.users = {}
        self.user_num = 0
        # 邻居网格
        self.n_side = 0  # neighbors的长度
        self.neighbors = []
        self.layers_neighbors_2d = []  # 二维index
        self.layers_neighbors_1d = []  # 一维index

    def get_node_index(self):
        return self._index

    def set_neighbors(self, nodes_list):
        self.neighbors = nodes_list
        self.n_side = len(nodes_list)

    def set_layers_neighbors_id(self, l_max, M, N, env):
        x, y = ids_1dto2d(self.get_node_index(), M, N)
        self.layers_neighbors_2d = get_layers_neighbors(x, y, l_max, M, N)  # 2维序号
        for layer_neighbors in self.layers_neighbors_2d:
            temp = []
            for item in layer_neighbors:
                x, y = item
                node_id = ids_2dto1d(x, y, M, N)
                if env.nodes[node_id] is not None:
                    temp.append(node_id)
            self.layers_neighbors_1d.append(temp)

    def add_courier(self, courier_id, courier):
        if courier_id not in self.couriers:
            self.couriers[courier_id] = courier
            self.online_courier_num += 1

    def add_shop(self, shop_id, shop):
        self.shops[shop_id] = shop
        self.shop_num += 1

    def add_user(self, user_id, user):
        self.users[user_id] = user
        self.user_num += 1

    def add_order(self, order):
        self.orders.append(order)
        self.order_num += 1

    def remove_unfinished_order(self, city_time):
        self.orders = []
        self.order_num = 0
