"""
owner: Zou ying Cao
data: 2023-02-01
description: 订单类
"""
from envs.route_planning import *


class Order(object):
    __slots__ = (
        "order_id", "shop_loc", "shop_latitude", "shop_longitude", "user_loc", "user_latitude", "user_longitude",
        "price", "real_reward", "order_create_time", "order_accept_time", "order_pickup_time",
        "order_delivery_time", "promise_delivery_duration", "real_delivery_duration",
        "flag", "overdue", "begin_p", "end_p")

    def __init__(self, order_id, shop_latitude, shop_longitude, user_latitude, user_longitude,
                 price, order_create_time, promise_delivery_duration, begin_p, end_p):

        self.order_id = order_id  # 订单编号

        self.shop_loc = Loc(shop_latitude, shop_longitude)  # 商家经纬度
        self.shop_longitude = shop_longitude
        self.shop_latitude = shop_latitude

        self.user_loc = Loc(user_latitude, user_longitude)  # 用户经纬度
        self.user_latitude = user_latitude
        self.user_longitude = user_longitude

        self.price = price  # 该订单用户支付的订单价格
        self.real_reward = 0  # 骑手的实际收益

        # 该单实际被平台接收的时间点:self.order_create_time
        self.order_create_time = order_create_time  # 传入的order_create_time为该单实际被用户创建的时间点
        self.order_accept_time = -1  # 该单实际被骑手接单的时间点
        self.order_pickup_time = -1  # 骑手实际到店取到该单的时间
        self.order_delivery_time = -1  # 骑手实际送达该单到用户的时间点

        self.promise_delivery_duration = promise_delivery_duration  # 平台给该订单送达所预估的配送时长(时间段,秒)
        self.real_delivery_duration = 0  # 实际配送时长(时间步)

        self.flag = 1   # 0：派送结束, 1：商家处, 2:用户处
        self.overdue = 0
        self.begin_p = begin_p
        self.end_p = end_p

    def set_shop_latitude(self, shop_latitude):
        self.shop_latitude = shop_latitude

    def set_shop_longitude(self, shop_longitude):
        self.shop_longitude = shop_longitude

    def set_user_latitude(self, user_latitude):
        self.user_latitude = user_latitude

    def set_user_longitude(self, user_longitude):
        self.user_longitude = user_longitude

    def set_order_accept_time(self, order_accept_time):
        self.order_accept_time = order_accept_time

    def set_order_pickup_time(self, order_pickup_time):
        self.order_pickup_time = order_pickup_time

    def set_order_delivery_time(self, order_delivery_time):
        self.order_delivery_time = order_delivery_time
        self.real_delivery_duration = self.order_delivery_time - self.order_accept_time

    def get_shop_latitude(self):
        return self.shop_latitude

    def get_shop_longitude(self):
        return self.shop_longitude

    def get_user_latitude(self):
        return self.user_latitude

    def get_user_longitude(self):
        return self.user_longitude

    def get_order_create_time(self):
        return self.order_create_time

    def get_order_accept_time(self):
        return self.order_accept_time

    def get_order_pickup_time(self):
        return self.order_pickup_time

    def get_order_delivery_time(self):
        return self.order_delivery_time

    def get_promise_delivery_duration(self):
        return self.promise_delivery_duration

    def cal_real_reward1(self, real_dur, dur):
        if real_dur <= dur:  # 未超时
            return self.price
        elif real_dur - dur < 1 * 5:  # 晚了1个时间步
            return 0.9 * self.price
        elif real_dur - dur < 2 * 5:  # 晚了2个时间步
            return 0.7 * self.price
        elif real_dur - dur < 3 * 5:  # 晚了3个时间步
            return 0.6 * self.price
        elif real_dur - dur < 4 * 5:  # 晚了4个时间步
            return 0.5 * self.price
        else:
            return 0.2 * self.price

    def cal_real_reward2(self, real_dur, dur):
        if real_dur <= dur:  # 未超时
            return self.price
        elif real_dur - dur < 1 * 5:  # 晚了1个时间步
            return 0.5 * self.price
        else:
            return 0

    def cal_duration(self):
        if self.order_delivery_time == -1 or self.order_accept_time == -1:
            pass
        self.real_delivery_duration = self.order_delivery_time - self.order_accept_time
        if self.real_delivery_duration > self.promise_delivery_duration:  # 超时
            self.overdue = 1
        self.real_reward = self.cal_real_reward2(self.real_delivery_duration, self.promise_delivery_duration)
        return self.real_delivery_duration, self.real_reward, self.overdue
