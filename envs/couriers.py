"""
owner: Zou ying Cao
data: 2023-02-01
description: 骑手类
"""
from envs.order import *


class Couriers(object):
    __slots__ = ("courier_id", "work_day_index", "work_day", "latitude", "longitude", "occur_time", "order_id",
                 "order_list", "num_order", "online", "full", "node", "record_time_slot", "capacity",
                 "shop_loc", "available_flag", "user_loc", "route_plan", "route_flag",
                 "next_stop", "next_time", "order_dict", "route",
                 "trip_start_time", "trip_end_time", "reset_lat", "reset_long", "reset_duration")

    def __init__(self, courier_id, latitude, longitude, time_slot_index):
        self.courier_id = courier_id
        self.work_day_index = 0
        self.work_day = []
        self.order_list = [[] for _ in range(30)]
        self.order_id = [[] for _ in range(30)]
        self.latitude = latitude
        self.longitude = longitude
        self.num_order = 0
        self.capacity = 10

        self.online = True
        self.full = False

        self.node = None
        self.occur_time = copy.deepcopy(time_slot_index)
        self.record_time_slot = copy.deepcopy(time_slot_index)

        self.available_flag = [[] for _ in range(30)]  # order.flag
        self.shop_loc = [[] for _ in range(30)]  # order对应商家经纬度
        self.user_loc = [[] for _ in range(30)]  # order对应用户经纬度
        self.route = None  # Route类
        self.route_plan = []  # 规划的路径中每一站经纬度
        self.route_flag = []  # self.route中available_loc对应的序号,self.route中available_loc=self.available_flag
        self.next_stop = 0  # 到达路径中的下一站的位置
        self.next_time = 0  # 到达路径中的下一站对应的route_flag的index
        # self.trip_start_time = 0
        # self.trip_end_time = 0
        # self.reset_lat = 0
        # self.reset_long = 0
        # self.reset_duration = 0

    def set_node(self, node):
        self.node = node

    def set_longitude(self, longitude):
        self.longitude = longitude

    def set_latitude(self, latitude):
        self.latitude = latitude

    def set_occur_time(self, city_time):
        self.occur_time = city_time

    def set_record_time_slot(self, city_time):
        self.record_time_slot = city_time

    def update_record_time_slot(self):
        self.record_time_slot += 1

    def get_state(self):
        state = [self.latitude, self.longitude, self.num_order, self.record_time_slot - self.occur_time]
        return state

    def clean_route(self):
        self.work_day.append(168 - self.occur_time)
        self.work_day_index += 1
        self.route = None
        self.route_plan = []
        self.route_flag = []
        self.next_stop = 0
        self.next_time = 0

    def clean_order_num(self):
        self.num_order = 0

    def step_update_state(self, env):
        # 骑手状态更新
        # 每一个时间步更新一次

        for i in range(self.next_time, len(self.route_flag)):
            t = self.route.time_list[i]  # 这一站到达时间
            lat = self.route.route_plan[i].latitude
            long = self.route.route_plan[i].longitude
            if t > self.record_time_slot:  # 还没到这一站
                if i == self.next_time:
                    k = 1 / (t - (self.record_time_slot - 1))  # self.record_time_slot - (self.record_time_slot -1)
                    self.latitude = (self.latitude - lat) * k + lat
                    self.longitude = (self.longitude - long) * k + long
                else:
                    k = (self.record_time_slot - self.route.time_list[i - 1]) / (t - self.route.time_list[i - 1])
                    self.latitude = (self.route.route_plan[i - 1].latitude - lat) * k + lat
                    self.longitude = (self.route.route_plan[i - 1].longitude - long) * k + long
                self.next_time = i
                break
            if self.available_flag[self.work_day_index][self.route_flag[i]] == 1:  # 已过这一站(商家处)
                self.available_flag[self.work_day_index][self.route_flag[i]] = 2
            elif self.available_flag[self.work_day_index][self.route_flag[i]] == 2:  # 完成一个订单派送(用户处)
                # self.order_list[self.work_day_index][self.route_flag[i]] = 0
                self.available_flag[self.work_day_index][self.route_flag[i]] = 0
                self.num_order -= 1
                if self.num_order < self.capacity:
                    self.full = False
        self.record_time_slot += 1

    def take_order(self, order, env):
        self.num_order += 1
        if self.num_order == self.capacity:
            self.full = True

        self.order_list[self.work_day_index].append(order)
        self.order_id[self.work_day_index].append(order.order_id)
        self.available_flag[self.work_day_index].append(order.flag)
        self.shop_loc[self.work_day_index].append(order.shop_loc)
        self.user_loc[self.work_day_index].append(order.user_loc)
        self.get_route_plan(env)

    def get_route_plan(self, env):
        order_id = copy.deepcopy(self.order_id[self.work_day_index])
        order_shop_loc = copy.deepcopy(self.shop_loc[self.work_day_index])
        order_user_loc = copy.deepcopy(self.user_loc[self.work_day_index])
        order_flag = copy.deepcopy(self.available_flag[self.work_day_index])

        self.route = Route(order_id, order_shop_loc, order_user_loc, Loc(self.latitude, self.longitude),
                           self.record_time_slot, order_flag)
        dis, self.route_plan, c_time, self.route_flag, _ = self.route.route_generation(env, self, True, order_id[-1])
        return dis, c_time

    def take_order_temp(self, order, env):
        o_l = copy.deepcopy(self.order_list[self.work_day_index])
        o_i = copy.deepcopy(self.order_id[self.work_day_index])
        a_l = copy.deepcopy(self.shop_loc[self.work_day_index])
        u_l = copy.deepcopy(self.user_loc[self.work_day_index])
        a_f = copy.deepcopy(self.available_flag[self.work_day_index])
        # 制作一个copy，实际上并没有改变真正的订单状况
        o_l.append(order)
        o_i.append(order.order_id)
        a_l.append(order.shop_loc)
        u_l.append(order.user_loc)
        a_f.append(order.flag)
        t_c, w_t, route = self.get_route_plan_temp(env, o_l, o_i, a_l, u_l, a_f)
        return t_c, w_t, route

    def get_route_plan_temp(self, env, o_l, o_i, a_l, u_l, a_f):
        # 计算新订单的加入后的额外时间
        route_plan_temp = Route(o_i, a_l, u_l, Loc(self.latitude, self.longitude),
                                self.record_time_slot, a_f)
        dis, route, c_time, route_flag, wait_time = route_plan_temp.route_generation(env, o_l, False, o_i[-1])
        if self.route:  # 原来的路线规划
            time_cost = c_time[-1] - self.route.time_list[-1]
        else:
            time_cost = c_time[-1] - self.record_time_slot
        route_ = []
        for r in route:
            route_.append(r.latitude)
            route_.append(r.longitude)
        for i in range(len(route_), self.capacity * 4, 2):  # 上限是六个订单：商家用户经纬度->4个数
            route_.append(route_[-2])
            route_.append(route_[-1])

        return time_cost, wait_time, route_
