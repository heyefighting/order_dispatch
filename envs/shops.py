"""
owner: Zou ying Cao
data: 2023-02-01
description:
"""
from envs.route_planning import *


class Shops(object):
    __slots__ = ("shop_id", "day_index", "order_list", "order_time_distance_radio", "longitude", "latitude", "location",
                 "order_num", "n_step", "node")

    def __init__(self, shop_id, shop_latitude, shop_longitude, time_slot=5):
        self.n_step = int(1440 / time_slot)
        self.day_index = 0
        self.shop_id = shop_id
        self.order_list = []  # need to clean in the end of day
        self.order_time_distance_radio = [[] for _ in range(30)]  # 30*1大小
        self.latitude = shop_latitude
        self.longitude = shop_longitude
        self.location = Loc(self.latitude, self.longitude)
        self.order_num = 0
        self.node = None

    def initialize_order_distribution(self, init_order):
        self.order_num = len(init_order)
        for i_order in init_order:
            if i_order.shop_latitude == self.latitude and i_order.shop_longitude == self.longitude:
                # self.order_list[i_order.order_accept_time].append(i_order)
                self.order_list.append(i_order)

    def set_node(self, node):
        self.node = node

    def set_longitude(self, longitude):
        self.longitude = longitude

    def set_latitude(self, latitude):
        self.latitude = latitude

    def increase_day_index(self):
        self.day_index += 1

    def clean_order_list(self):
        self.order_list.clear()

    def cal_time_distance_ratio(self):
        temp_radio_list = []
        for i_order in self.order_list:
            dist = get_distance_hav(i_order.user_latitude, i_order.user_longitude, self.latitude, self.longitude)
            if i_order.get_order_delivery_time() != -1 and i_order.get_order_pickup_time() != -1:
                t = i_order.get_order_delivery_time() - i_order.get_order_pickup_time()
            else:
                # t = i_order.get_promise_delivery_duration()
                continue
            if dist == 0:
                radio = 0
            else:
                radio = t / dist
            temp_radio_list.append(radio)
        self.order_time_distance_radio[self.day_index] = [sum(temp_radio_list)]

    def add_one_order_time_distance_ratio(self, order):
        distance = get_distance_hav(order.user_latitude, order.user_longitude, self.latitude, self.longitude)
        if order.get_order_delivery_time() != -1 and order.get_order_pickup_time() != -1:
            t = order.order_delivery_time - order.order_pickup_time
        else:
            t = order.get_promise_delivery_duration()
        radio = t / distance
        # self.order_time_distance_radio[order.get_order_pickup_time()].append(radio)
        self.order_time_distance_radio[self.day_index].append(radio)

    def add_order(self, order):
        self.order_list.append(order)
        self.order_num += 1
