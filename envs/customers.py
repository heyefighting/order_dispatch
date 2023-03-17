"""
owner: Zou ying Cao
data: 2023-02-01
description: 顾客类
"""
from envs.route_planning import *


class Customers(object):
    __slots__ = ("user_id", "order_list", "order_wait_time", "longitude", "latitude", "location",
                 "order_num", "n_step", "node")

    def __init__(self, user_id, user_latitude, user_longitude, time_slot=5):
        self.n_step = int(1440 / time_slot)
        self.user_id = user_id
        # self.order_list = [[] for _ in range(self.n_step)]
        # self.order_wait_time = [[] for _ in range(self.n_step)]
        self.order_list = []
        self.order_wait_time = []
        self.latitude = user_latitude
        self.longitude = user_longitude
        self.location = Loc(self.latitude, self.longitude)
        self.order_num = 0
        self.node = None

        # self.initialize_order_creation(init_order)
        # self.construct_wait_time()

    def initialize_order_creation(self, init_order):
        self.order_num = len(init_order)
        for i_order in init_order:
            if i_order.user_latitude == self.latitude and i_order.user_longitude == self.longitude:
                self.order_list[i_order.order_create_time].append(i_order)

    def set_node(self, node):
        self.node = node

    def set_longitude(self, longitude):
        self.longitude = longitude

    def set_latitude(self, latitude):
        self.latitude = latitude

    def construct_wait_time_old(self):
        self.order_wait_time = [[] for _ in range(self.n_step)]
        for i in range(len(self.order_list)):
            for order in self.order_list[i]:
                if order.order_delivery_time != -1:
                    wait_time = order.order_delivery_time - order.order_create_time \
                                - order.get_promise_delivery_duration()
                else:
                    wait_time = order.get_promise_delivery_duration()
                self.order_wait_time[order.order_create_time].append(wait_time)

    def construct_wait_time(self):
        self.order_wait_time = []
        for order in self.order_list:
            if order.order_delivery_time != -1:
                wait_time = order.order_delivery_time - order.order_create_time \
                                - order.get_promise_delivery_duration()
            else:
                wait_time = order.get_promise_delivery_duration()
            self.order_wait_time.append(wait_time)

    def add_wait_time(self, order):
        wait_time = order.order_delivery_time - order.order_create_time - order.get_promise_delivery_duration()
        # self.order_wait_time[order.order_create_time].append(wait_time)
        self.order_wait_time.append(wait_time)

    def create_order(self, order):  # 下单
        self.order_list.append(order)
        self.order_num += 1
