"""
owner: Zou ying Cao
data: 2023-02-01
description: 骑手的路径规划
"""
import copy
from env_utils.utils import *


class Loc:

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def __eq__(self, other):
        return self.latitude == other.latitude and self.longitude == other.longitude

    def __hash__(self):
        return hash((self.latitude, self.latitude))  # str(self.latitude)+str(self.latitude)


class Route:

    def __init__(self, order_id, shop_loc, user_loc, courier_loc, timestamp, available_flag):
        self.order_id = order_id
        self.shop_loc = copy.deepcopy(shop_loc)
        self.user_loc = user_loc
        self.courier_loc = courier_loc
        self.time_duration = timestamp
        self.route_plan = []
        self.route_distance = 0
        self.available_loc = copy.deepcopy(shop_loc)
        self.available_flag = available_flag  # 用来标记路径规划
        self.route_flag = []
        self.time_list = []  # 每一个位置结束时的时间
        self.cost_time = 0

    def next_stop_generation(self, avail_loc, avail_flag):
        # 计算每个点与当前点的距离
        _distance = []
        for i in range(len(avail_loc)):
            if avail_flag[i] != 0:
                _distance.append((i, get_distance_hav(self.courier_loc.latitude, self.courier_loc.longitude,
                                                      avail_loc[i].latitude, avail_loc[i].longitude)))

        _distance.sort(key=lambda x: x[1])  # x[1]表示用第二个元素即距离排序
        min_loc_id = _distance[0][0]
        min_dis = _distance[0][1]

        self.time_duration = self.time_duration + min_dis / 25 * 12  # 25km/h, 小数
        self.time_list.append(self.time_duration)

        self.courier_loc = avail_loc[min_loc_id]  # 骑手移到下一站位置
        return min_loc_id, min_dis

    def route_generation(self, env, courier, flag, new_id):
        # 每次选最近的地点
        # flag==true:courier为courier
        # flag==false:courier为order_list
        wait_time = None
        while sum(self.available_flag) != 0:  # 还有订单
            loc_id, loc_dis = self.next_stop_generation(self.available_loc, self.available_flag)  # 找最近
            self.route_plan.append(self.available_loc[loc_id])  # 下一站（经纬度）
            self.route_flag.append(loc_id)  # 对应available_loc的序号
            self.route_distance += loc_dis
            if self.available_flag[loc_id] == 1:  # 下一站是商家
                if flag:
                    courier.order_list[loc_id].set_order_pickup_time(self.time_list[-1])
                    env.step_shop_state_update(self.shop_loc[loc_id], self.user_loc[loc_id],
                                               self.time_list[-1], self.order_id[loc_id])  # 找到对应的商家、用户更新
                self.available_loc[loc_id] = self.user_loc[loc_id]  # 只有取到餐才能去送给用户，这时用户的经纬度才能放到available_loc中
                self.available_flag[loc_id] = 2
            elif self.available_flag[loc_id] == 2:  # 下一站是用户
                if flag:
                    courier.order_list[loc_id].set_order_delivery_time(self.time_list[-1])
                    env.step_user_state_update(self.shop_loc[loc_id], self.user_loc[loc_id],
                                               self.time_list[-1], self.order_id[loc_id])
                else:
                    if new_id == self.order_id[loc_id]:  # 实际送达时间-预期送达时间
                        wait_time = (self.time_list[-1] - courier[loc_id].get_order_create_time()) * 5 - \
                                   courier[loc_id].get_promise_delivery_duration()
                self.available_flag[loc_id] = 0  # 对应的经纬度在下次路线规划时不会考虑
                self.available_loc[loc_id] = 0
            elif self.available_flag[loc_id] == 0:  # 应该没有这个情况
                pass
        self.cost_time = self.time_list[-1]

        return self.route_distance, self.route_plan, self.time_list, self.route_flag, wait_time
