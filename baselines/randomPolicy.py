"""
owner: Zou ying Cao
data: 2023-04-01
description: random policy
"""
import random
from monitor import Monitor
from torch.nn import MSELoss

from envs.envs import Environment
from env_utils.neighbours import *
from envs.order_dispatching import *

save_dir = 'save-dir'


class randomPolicy:
    def __init__(self, simulator, Monitor, print_updates=True):
        self.env = simulator

        # configs
        self.episode_num = 0

        self.mse_loss = MSELoss(reduction='mean')
        self.print_updates = print_updates
        self.monitor = Monitor

        self.mean_rewards = []
        self.mean_costs = []
        self.mean_value_loss = []
        self.mean_cost_loss = []

    def take_action(self, len):
        c_id = random.choice(list(range(len)))
        return c_id

    def train(self, n_episodes, n_step, alpha, beta):
        trajectory_value_loss = []
        trajectory_cost_loss = []
        while self.episode_num < n_episodes:
            fileName = "../datasets/orderData" + str(self.episode_num + 1) + ".csv"
            orders_data = pd.read_csv(fileName)
            self.env.reset_env(orders_data)  # 出现新的订单与用户

            trajectory_value_loss.clear()
            trajectory_cost_loss.clear()
            dispatch_action = {}
            trajectory_rewards = []
            trajectory_costs = []

            for i_step in range(1, n_step + 1):
                dispatch_action.clear()
                trajectory_rewards.clear()
                trajectory_costs.clear()
                for i_order in self.env.day_orders[self.env.time_slot_index]:
                    self.env.users_dict[i_order.user_loc].create_order(i_order)  # 用户下单
                    self.env.set_node_one_order(i_order)  # 商家所在路网节点订单+1
                    state = self.env.get_region_state()  # 路网中骑手、用户、商家、订单分布(其中商家分布不变)
                    shop_node_id = i_order.begin_p  # 订单所在商家的对应node_id
                    x, y = ids_1dto2d(shop_node_id, self.env.M, self.env.N)
                    order_num = state[2][x][y]  # 附近的订单个数

                    id_list, couriers, wait_time, action = self.env.action_collect(x, y, i_order, order_num)
                    if len(couriers) == 0:  # 8邻域无可用的骑手, 多一层邻域查找
                        for layer in range(6):
                            id_list, couriers, wait_time, action = self.env.more_action_collect(layer + 2, x, y,
                                                                                                i_order, order_num)
                            if len(couriers):
                                break
                    if len(couriers):
                        state_couriers = get_courier_state(self.env, state, couriers)
                        state_input = get_state_input(state_couriers, action)  # state, action拼接
                        c_id = self.take_action(len(state_input))  # random

                        i_order.set_order_accept_time(self.env.time_slot_index)  # 订单设置接单时间
                        self.env.shops_dict[i_order.shop_loc].add_order(i_order)  # 相应商家order_list添加该订单
                        # courier部分状态更新包含num_order与接待订单的时间点属性
                        self.env.couriers_dict[id_list[c_id]].take_order(i_order, self.env)
                        self.env.nodes[i_order.begin_p].order_num -= 1

                        d = DispatchPair(i_order, self.env.couriers_dict[id_list[c_id]])
                        d.set_state_input(state_input)  # [c_id]
                        d.set_state(state_couriers[c_id])  #
                        d.set_action(c_id)  # int
                        record = False
                        if i_step == n_step and self.env.time_slot_index == 167:
                            record = True
                        d.set_reward(self.env, alpha, beta, self.episode_num, record)
                        d.set_cost(wait_time[c_id])  # 0:不超时, 1:超时
                        trajectory_rewards.append(d.reward)
                        trajectory_costs.append(d.cost)
                        dispatch_action[i_order] = d
                    else:
                        if self.env.time_slot_index != 167:
                            print('one order late')
                            # d = DispatchPair
                            # d.set_reward(self.env, alpha, beta, self.episode_num, True)
                            self.env.day_orders[self.env.time_slot_index + 1].append(i_order)  # 放到下一时刻再分配
                        else:
                            print('one order dismissed')

                if len(dispatch_action):
                    dispatch_result = self.env.step(dispatch_action)  # state为路网的next_state

                    self.mean_rewards.append(torch.mean(torch.Tensor(trajectory_rewards)))
                    self.mean_costs.append(torch.mean(torch.Tensor(trajectory_costs)))
                    self.monitor.update_reward(self.episode_num * n_step + i_step, self.mean_rewards[-1])
                    self.monitor.update_cost(self.episode_num * n_step + i_step, self.mean_costs[-1])

            self.episode_num += 1
            self.env.update_env()
            if self.print_updates:
                self.print_update()

    def print_update(self):
        update_message = '[Episode]: {0} | [Avg. Reward]: {1} | [Avg. Cost]: {2}'
        format_args = (self.episode_num, self.mean_rewards[-1], self.mean_costs[-1])
        print(update_message.format(*format_args))


if __name__ == '__main__':
    # setup the environment
    shop_data = pd.read_csv("../shopInitialization.csv")  # 从30天数据中提取出来的相关商家（包括经纬度、所在路网网格ID）
    mapped_matrix_int = np.arange(0, 100, 1).reshape([10, 10])  # 用一个矩形来网格化配送地图，不可送达地（湖泊、海洋等）标记-1
    environment = Environment(shop_data, mapped_matrix_int, 10, 10)
    monitor = Monitor(train=True, spec="CMO_RL_Dispatch")

    policy = randomPolicy(simulator=environment, Monitor=monitor)

    # for train
    n_episodes = 20
    n_steps = 168  # 一天内
    alpha = 0.3  # reward中骑手公平的权重
    beta = 0.3  # reward中商家公平的权重

    policy.train(n_episodes, n_steps, alpha, beta)
