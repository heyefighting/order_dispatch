"""
owner: Zou ying Cao
data: 2023-02-01
description: simulation of instant delivery order dispatch scenario
"""
from envs.nodes import *
from envs.shops import *
from envs.couriers import *
from envs.customers import *


class Environment:
    __slots__ = (
        "M", "N", "n_valid_nodes", "valid_node_1d", "nodes", "state", "n_step",
        "time_slot_index", "day_orders", "n_couriers", "couriers_dict", "day_couriers",
        "shops_init", "n_shops", "shops_dict", "users_init", "n_users", "users_dict")

    def __init__(self, shop_data, valid_matrix, M, N, courier_num=4000):
        self.M = M
        self.N = N
        self.n_valid_nodes = M * N  # 有效配送的网格个数
        self.valid_node_1d = []  # 一维序号, 例：10*10网格中，(1,1)->11
        self.nodes = [Node(i) for i in range(M * N)]  # 路网划分后对应的所有node，一维

        self.state = np.zeros((3, self.M, self.N))  # 路网骑手、商家、订单分布
        self.n_step = 168  # 早上8点到晚上10点，每五分钟一个时间区间，从0到167
        self.time_slot_index = 0  # 时间步记录

        self.day_orders = [[] for _ in range(self.n_step)]  # 订单按时间步分类

        self.n_couriers = courier_num
        self.couriers_dict = {}  # key为骑手id, 编号为1~courier_num
        self.day_couriers = [[] for _ in range(self.n_step)]  # 每一个时间步在线的骑手集合

        self.shops_init = []  # 从shop_data提取出的shops, 在initialize_shops中初始化
        self.n_shops = 0
        self.shops_dict = {}  # key为商家location

        self.users_init = []  # 在initialize_users中初始化
        self.n_users = 0
        self.users_dict = {}  # key为用户location

        self.initialize_couriers(courier_num)  # 根据courier_num初始化骑手,couriers_dict,day_couriers
        self.initialize_shops(shop_data)  # 从数据读取初始化商家 shops_dict
        self.set_valid_node(valid_matrix)  # 初始化有效节点,valid_node_1d,n_valid_nodes,nodes
        self.set_node_neighbors(8)  # 设置节点8邻居node集合
        self.set_node_shops(self.shops_init)  # 用商家位置索引self.shops_dict，所有商家都包含

    def initialize_couriers(self, courier_num):
        for i in range(courier_num):
            lat = np.random.uniform(31.2, 31.3)
            lon = np.random.uniform(121.4, 121.5)
            occur_time = 0
            if i >= 3000:
                occur_time = (self.time_slot_index + i) % 24
            c = Couriers(int(i + 1), lat, lon, occur_time)
            c.set_node(self.nodes[gps_to_id(lat, lon)])
            self.couriers_dict[c.courier_id] = c
            self.day_couriers[c.occur_time].append(c)

    def update_couriers(self):
        for _, i_courier in self.couriers_dict.items():
            lat = np.random.uniform(31.2, 31.3)
            lon = np.random.uniform(121.4, 121.5)
            self.couriers_dict[_].set_latitude(lat)
            self.couriers_dict[_].set_longitude(lon)
            self.couriers_dict[_].clean_order_num()
            self.couriers_dict[_].clean_route()
            occur_time = 0
            if _ >= 3000:
                occur_time = (self.time_slot_index + _) % 24
            self.couriers_dict[_].set_occur_time(occur_time)
            self.couriers_dict[_].set_record_time_slot(occur_time)
            self.couriers_dict[_].set_node(self.nodes[gps_to_id(lat, lon)])

    def initialize_shops(self, data):
        self.shops_init = []
        for i in range(len(data)):
            s = Shops(int(i + 1),
                      data['merchant_latitude'].tolist()[i], data['merchant_longitude'].tolist()[i])
            s.set_node(self.nodes[int(data['arrival_id'].tolist()[i])])
            self.shops_init.append(s)
            self.shops_dict[s.location] = s
        self.n_shops = len(self.shops_init)

    def update_shops(self):
        for _, shop in self.shops_dict.items():
            self.shops_dict[_].increase_day_index()

    def initialize_orders(self, orders_data):
        self.day_orders = [[] for _ in range(self.n_step)]
        for i in range(len(orders_data)):
            order = Order(int(i + 1),
                          orders_data['merchant_latitude'].tolist()[i], orders_data['merchant_longitude'].tolist()[i],
                          orders_data['user_latitude'].tolist()[i], orders_data['user_longitude'].tolist()[i],
                          orders_data['delivery_fee'].tolist()[i],  # price
                          int(orders_data['time'].tolist()[i]),  # 创建时间
                          int(orders_data['promise_time'].tolist()[i]),
                          int(orders_data['arrival_id'].tolist()[i]),
                          int(orders_data['delivery_id'].tolist()[i]))
            self.day_orders[int(orders_data['time'].tolist()[i])].append(order)

    def initialize_users(self, orders_data):
        self.users_init = []
        data = orders_data.drop_duplicates(subset=['user_latitude', 'user_longitude'], keep='first')
        data.reset_index(drop=True)
        for i in range(len(data)):
            user = Customers(int(i + 1),
                             data['user_latitude'].tolist()[i], data['user_longitude'].tolist()[i])
            user.set_node(self.nodes[int(data['delivery_id'].tolist()[i])])
            self.users_dict[user.location] = user
            self.users_init.append(user)
        self.n_users = len(self.users_init)

    def reset_env(self, orders_data):
        self.initialize_orders(orders_data)  # 从数据读取初始化订单 day_orders
        self.initialize_users(orders_data)  # 从数据读取初始化用户 users_init,users_dict, n_users
        self.set_node_couriers(self.day_couriers[self.time_slot_index])  # 当前时间步, 每个路网网格的在线骑手更新
        # self.set_node_order(self.day_orders[self.time_slot_index])  # 每个路网网格的累积订单更新
        self.set_node_users(self.users_init)  # 每个路网网格的累积用户更新

    def update_env(self):
        self.time_slot_index = 0
        self.update_couriers()  # 重置courier_dict中骑手新的一天上线时间、出现位置、清空路线、工作天数+1
        # self.initialize_couriers(self.n_couriers)  # 骑手收入不再累积, 故初始化
        self.update_shops()  # 商家运营天数+1
        self.generate_one_day_couriers()  # 按照重置的上线时间，重置self.day_couriers

    def generate_one_day_couriers(self):  # 按照重置的上线时间初始化self.day_couriers
        day_couriers = [[] for _ in range(self.n_step)]
        for _, i_courier in self.couriers_dict.items():
            day_couriers[i_courier.occur_time].append(i_courier)
        self.day_couriers = day_couriers

    def set_node_order(self, day_order):
        for i_order in day_order:
            shop_id = i_order.begin_p
            self.nodes[shop_id].add_order(i_order)

    def set_node_one_order(self, order):
        shop_id = order.begin_p
        self.nodes[shop_id].add_order(order)

    def set_node_couriers(self, day_couriers):
        if len(day_couriers):
            for i_courier in day_couriers:
                self.nodes[i_courier.node.get_node_index()].add_courier(i_courier.courier_id, i_courier)

    def set_node_shops(self, shops_init):
        for shop in shops_init:
            self.nodes[shop.node.get_node_index()].add_shop(shop.shop_id, shop)

    def set_node_users(self, users_init):
        for user in users_init:
            # self.users_dict[user.location] = user
            self.nodes[user.node.get_node_index()].add_user(user.user_id, user)

    def get_region_state(self):
        for _node in self.nodes:
            if _node is not None:
                row_id, column_id = ids_1dto2d(_node.get_node_index(), self.M, self.N)
                self.state[0, row_id, column_id] = _node.online_courier_num
                self.state[1, row_id, column_id] = _node.shop_num
                self.state[2, row_id, column_id] = _node.order_num
        return self.state

    def set_valid_node(self, mapped_matrix_int):  # 初始化节点，当然只有有效节点能够被初始化
        row_index, col_index = np.where(mapped_matrix_int >= 0)  # 有效节点的行坐标数组和列坐标数组
        valid_ids = []
        for x, y in zip(row_index, col_index):  # 用zip()将行列坐标打包成元祖
            node_id = ids_2dto1d(x, y, self.M, self.N)
            self.nodes[node_id] = Node(node_id)
            valid_ids.append(node_id)
            self.nodes[node_id].set_layers_neighbors_id(1, self.M, self.N, self)  # 八邻域，更改：2->1层
        self.valid_node_1d = valid_ids
        self.n_valid_nodes = len(valid_ids)

    def set_node_neighbors(self, n_side):
        for idx, current_node in enumerate(self.nodes):
            if current_node is not None:  # 排除非有效派送网格区域
                i, j = ids_1dto2d(idx, self.M, self.N)
                self.nodes[idx].set_neighbors(get_neighbor_list(i, j, self.M, self.N, n_side, self.nodes))

    def step_increase_time_slot_index(self):
        self.time_slot_index += 1
        self.time_slot_index = self.time_slot_index % self.n_step
        for courier_id, courier in self.couriers_dict.items():
            self.couriers_dict[courier_id].set_record_time_slot(self.time_slot_index)

    def step_couriers_state_update(self):  # 骑手状态更新，每一个时间步更新一次
        for courier_id, i_courier in self.couriers_dict.items():
            if i_courier.occur_time < self.time_slot_index:
                self.couriers_dict[courier_id].step_update_state(self)
            if i_courier.occur_time < self.time_slot_index and i_courier.online is True:
                self.day_couriers[self.time_slot_index].append(i_courier)

    def step_shop_state_update(self, shop_loc, user_loc, pickup_time, order_id):
        for index, i_order in enumerate(self.shops_dict[shop_loc].order_list[self.shops_dict[shop_loc].day_index]):
            if i_order.order_id == order_id:
                self.shops_dict[shop_loc].order_list[self.shops_dict[shop_loc].day_index][index]. \
                    set_order_pickup_time(pickup_time)
                break
        for index, i_order in enumerate(self.users_dict[user_loc].order_list):
            if i_order.order_id == order_id:
                self.users_dict[user_loc].order_list[index].set_order_pickup_time(pickup_time)
                break

    def step_user_state_update(self, shop_loc, user_loc, deli_time, order_id):
        for index, i_order in enumerate(self.shops_dict[shop_loc].order_list[self.shops_dict[shop_loc].day_index]):
            if i_order.order_id == order_id:
                self.shops_dict[shop_loc].order_list[self.shops_dict[shop_loc].day_index][index]. \
                    set_order_delivery_time(deli_time)
                self.shops_dict[shop_loc].cal_time_distance_ratio()
                break
        for index, i_order in enumerate(self.users_dict[user_loc].order_list):
            if i_order.order_id == order_id:
                self.users_dict[user_loc].order_list[index].set_order_delivery_time(deli_time)
                self.users_dict[user_loc].construct_wait_time()
                break

    def user_fairness_before_action(self, user_loc, delivery_time, order_id):
        for index, i_order in enumerate(self.users_dict[user_loc].order_list):
            if i_order.order_id == order_id:
                return (delivery_time - self.users_dict[user_loc].order_list[index].get_order_create_time()) \
                       * 5 - self.users_dict[user_loc].order_list[index].get_promise_delivery_duration()

    def step(self, dispatch_action):
        self.step_increase_time_slot_index()  # time_slot_index+1
        # self.set_node_order(self.day_orders[self.time_slot_index])
        self.set_node_couriers(self.day_couriers[self.time_slot_index])  # 更新node的online_courier_num

        self.state = self.get_region_state()
        self.step_couriers_state_update()  # 骑手状态更新

        if len(dispatch_action) != 0:
            for order, dispatch_pair in dispatch_action.items():
                courier = dispatch_pair.courier
                next_state = courier.get_state()  # [latitude, longitude, num_order, record_time_slot]
                dispatch_action[order].set_next_state(next_state)

        return dispatch_action

    def action_collect(self, x, y, order, order_num):
        couriers = []
        actions = []
        wait_times = []
        id_list = []
        neighbours = get_neighbor_list(x, y, self.M, self.N, 8, self.nodes)
        for i in range(8):  # 八邻域
            if neighbours[i] is not None:
                for courier_id, ic in neighbours[i].couriers.items():
                    if ic.full is False and ic.online is True and ic.occur_time <= self.time_slot_index:
                        shop_lat = order.shop_latitude
                        shop_long = order.shop_longitude
                        user_lat = order.user_latitude
                        user_long = order.user_longitude
                        distance = get_distance_hav(ic.latitude, ic.longitude, shop_lat, shop_long)

                        time_cost, wait_time, route = ic.take_order_temp(order, self)

                        id_list.append(ic.courier_id)  # 骑手编号
                        couriers.append(ic.get_state())  # 骑手状态
                        wait_times.append(wait_time)  # 骑手当前单+该订单后：（每一单）实际送达时间-预计到达时间

                        action = [shop_lat, shop_long, user_lat, user_long, distance, time_cost, order_num]
                        action.extend(route)
                        actions.append(action)

        return id_list, couriers, wait_times, actions

    def more_action_collect(self, layer, x, y, order, order_num):
        couriers = []
        actions = []
        wait_times = []
        id_list = []
        neighbours = get_layers_neighbors(x, y, layer, self.M, self.N)
        for i in range(layer):
            for item in neighbours[i]:
                xx, yy = item
                node_id = ids_2dto1d(xx, yy, self.M, self.N)
                nodes = self.nodes[node_id]
                if nodes is not None:
                    for courier_id, ic in nodes.couriers.items():
                        if ic.full is False and ic.online is True and ic.occur_time <= self.time_slot_index:
                            shop_lat = order.shop_latitude
                            shop_long = order.shop_longitude
                            user_lat = order.user_latitude
                            user_long = order.user_longitude
                            distance = get_distance_hav(ic.latitude, ic.longitude, shop_lat, shop_long)

                            time_cost, wait_time, route = ic.take_order_temp(order, self)

                            id_list.append(ic.courier_id)  # 骑手编号
                            couriers.append(ic.get_state())  # 骑手状态
                            wait_times.append(wait_time)  # 骑手当前单+该订单后：（每一单）实际送达时间-预计到达时间

                            action = [shop_lat, shop_long, user_lat, user_long, distance, time_cost, order_num]
                            action.extend(route)
                            actions.append(action)

        return id_list, couriers, wait_times, actions
