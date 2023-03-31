"""
owner: Zou ying Cao
data: 2023-03-26
description: test the memory
"""
import random
import pandas as pd
from monitor import Monitor
from torch.nn import MSELoss
from models import Actor, CostCritic, ValueCritic

from envs.envs import Environment
from env_utils.neighbours import *
from envs.order_dispatching import *
from optimization_utils.hvp import get_Hvp_fun
from optimization_utils.line_search import line_search
from optimization_utils.conjugate_gradient import cg_solver
from torch_utils.distribution_utils import mean_kl_first_fixed
from torch_utils.torch_utils import flat_grad, get_flat_params, set_params

save_dir = 'save-dir'


def discount(value, discount_term):
    n = value.size(0)
    disc_pow = torch.pow(discount_term, torch.arange(n).float())
    reverse_index = torch.arange(n - 1, -1, -1)

    discounted = torch.cumsum((value * disc_pow)[reverse_index], dim=-1)[reverse_index] / disc_pow

    return discounted


# 广义优势估计:A^(GAE)_t=\sum(gamma*lambda)^l*delta_(t+l)
def compute_advantage(actual_value, exp_value, discount_term, bias_red_param):
    exp_value_next = torch.cat([exp_value[1:], torch.tensor([0.0])])
    td_res = actual_value + discount_term * exp_value_next - exp_value  # 时序差分误差td_error:r+gamma*V(s_t+1)-V(s_t)
    advantage = discount(td_res, discount_term * bias_red_param)

    return advantage


# GAE的代码
def compute_adv(gamma, lm, td_delta):
    """
    GAE 将不同步数的优势估计进行指数加权平均
    :param gamma: 折扣因子
    :param lm:
           lm=0时, A_t=delta_t=r+gamma*V(s_t+1)-V(s_t),仅仅只看一步差分得到的优势;
           lm=1时, A_t=sum_{l=0}^{T}(gamma^l*r_{t+l})是看每一步差分得到优势的完全平均值
    :param td_delta: 时序差分
    :return:优势函数
    """
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lm * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float32)


class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = None
        self.state_inputs = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.cost = None
        # self.policy = None

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, sa, s, a, r, next_s, c):
        if self.curr_lens == 0:
            self.states = s
            self.state_inputs = sa
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = len(self.states)
            self.cost = c
            # self.policy = p

        elif self.curr_lens + len(s) <= self.memory_size:  # len(s)=s.shape[0]
            # self.states = np.concatenate((self.states, s), axis=0)
            # self.state_inputs = np.concatenate((self.state_inputs, sa), axis=0)
            # self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            # self.actions = np.concatenate((self.actions, a), axis=0)
            # self.rewards = np.concatenate((self.rewards, r), axis=0)
            # self.curr_lens = self.states.shape[0]
            # self.cost = np.concatenate((self.cost, c), axis=0)
            # self.policy = np.concatenate((self.policy, p), axis=0)
            self.states.extend(s)
            self.state_inputs.extend(sa)
            self.next_states.extend(next_s)
            self.actions.extend(a)
            self.rewards.extend(r)
            self.curr_lens = len(self.states)
            self.cost.extend(c)
        else:
            new_sample_lens = len(s)
            reserve_lens = self.memory_size - new_sample_lens

            self.states[0:reserve_lens] = self.states[self.curr_lens - reserve_lens:self.curr_lens]
            self.state_inputs[0:reserve_lens] = self.state_inputs[self.curr_lens - reserve_lens:self.curr_lens]
            self.actions[0:reserve_lens] = self.actions[self.curr_lens - reserve_lens:self.curr_lens]
            self.rewards[0:reserve_lens] = self.rewards[self.curr_lens - reserve_lens:self.curr_lens]
            self.next_states[0:reserve_lens] = self.next_states[self.curr_lens - reserve_lens:self.curr_lens]
            self.cost[0:reserve_lens] = self.cost[self.curr_lens - reserve_lens:self.curr_lens]
            # self.policy[0:reserve_lens] = self.policy[self.curr_lens - reserve_lens:self.curr_lens]

            self.states[self.curr_lens: self.memory_size] = s
            self.state_inputs[self.curr_lens: self.memory_size] = sa
            self.actions[self.curr_lens: self.memory_size] = a
            self.rewards[self.curr_lens: self.memory_size] = r
            self.next_states[self.curr_lens: self.memory_size] = next_s
            self.cost[self.curr_lens: self.memory_size] = c
            # self.policy[self.curr_lens: self.memory_size] = p

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.state_inputs, self.states, self.rewards, self.next_states, self.cost, self.actions]
        # indices = random.sample(range(0, self.curr_lens), self.batch_size)
        rand = random.randint(0, self.curr_lens - self.batch_size)
        batch_s = self.states[rand:rand + self.batch_size]
        batch_sa = self.state_inputs[rand:rand + self.batch_size]
        batch_a = self.actions[rand:rand + self.batch_size]
        batch_r = self.rewards[rand:rand + self.batch_size]
        batch_next_s = self.next_states[rand:rand + self.batch_size]
        batch_c = self.cost[rand:rand + self.batch_size]
        # batch_p = self.policy[indices]
        return batch_sa, batch_s, batch_r, batch_next_s, batch_c, batch_a


class CPO:
    def __init__(self, state_dim, action_dim, capacity, batch_size, simulator, device, Monitor, max_kl=1e-2,
                 val_lr=1e-2, cost_lr=1e-2, max_constraint_val=0.1, val_small_loss=1e-3, cost_small_loss=1e-3,
                 discount_val=0.995, discount_cost=0.995, lambda_val=0.98, lambda_cost=0.98,
                 line_search_coefficient=0.9, line_search_max_iter=10, line_search_accept_ratio=0.1,
                 continue_from_file=True, save_every=1, print_updates=True):
        # Actor
        self.policy = Actor(state_dim, action_dim)
        # Critic for value function
        self.value_function = ValueCritic(state_dim)
        # Critic for cost function
        self.cost_function = CostCritic(state_dim)
        # replay buffer
        self.replay = ReplayMemory(capacity, batch_size)
        # environment for order dispatching
        self.env = simulator

        # configs
        self.episode_num = 0
        self.batch_size = batch_size

        self.max_kl = max_kl
        self.max_constraint_val = max_constraint_val

        self.val_small_loss = val_small_loss
        self.cost_small_loss = cost_small_loss
        self.discount_val = discount_val
        self.discount_cost = discount_cost
        self.lambda_val = lambda_val
        self.lambda_cost = lambda_cost

        self.line_search_coefficient = line_search_coefficient
        self.line_search_max_iter = line_search_max_iter
        self.line_search_accept_ratio = line_search_accept_ratio

        self.mse_loss = MSELoss(reduction='mean')
        self.value_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=val_lr)
        self.cost_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=cost_lr)

        self.device = device
        self.save_every = save_every
        self.print_updates = print_updates
        self.monitor = Monitor

        self.mean_rewards = []
        self.mean_costs = []
        self.mean_value_loss = []
        self.mean_cost_loss = []

        if continue_from_file:
            self.load_session()

    def take_action(self, state, epsilon):
        state_ = torch.tensor(state, dtype=torch.float32)
        action_prob = self.policy(state_)
        if np.random.random() < epsilon:
            c_id = torch.distributions.Categorical(action_prob).sample().item()
        else:
            maxi = max(action_prob)
            c_id = random.choice([index for index, t in enumerate(action_prob) if t == maxi])
        return c_id

    def train(self, n_episodes, n_step, alpha, beta):
        trajectory_value_loss = []
        trajectory_cost_loss = []
        while self.episode_num < n_episodes:
            fileName = "datasets/orderData" + str(self.episode_num + 1) + ".csv"
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
                        # c_id = self.take_action(state_input)  # , softmax_V
                        c_id = self.take_action(state_input, 0.02)  # , softmax_V

                        i_order.set_order_accept_time(self.env.time_slot_index)  # 订单设置接单时间
                        self.env.shops_dict[i_order.shop_loc].add_order(i_order)  # 相应商家order_list添加该订单
                        # courier部分状态更新包含num_order与接待订单的时间点属性
                        self.env.couriers_dict[id_list[c_id]].take_order(i_order, self.env)
                        self.env.nodes[i_order.begin_p].order_num -= 1

                        d = DispatchPair(i_order, self.env.couriers_dict[id_list[c_id]])
                        d.set_state_input(state_input)  # [c_id]
                        d.set_state(state_couriers[c_id])  #
                        d.set_action(c_id)  # int
                        d.set_reward(self.env, alpha, beta)
                        d.set_cost(wait_time[c_id])  # 0:不超时, 1:超时
                        trajectory_rewards.append(d.reward)
                        trajectory_costs.append(d.cost)
                        dispatch_action[i_order] = d
                    else:
                        if self.env.time_slot_index != 167:
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

    def load_session(self):
        actor_load_path = os.path.join(save_dir, 'model0329_actor' + '.pt')
        actor_ptFile = torch.load(actor_load_path)
        cost_load_path = os.path.join(save_dir, 'model0330_critic' + '.pt')
        cost_ptFile = torch.load(cost_load_path)

        self.policy.load_state_dict(actor_ptFile['policy_state_dict'])
        self.value_function.load_state_dict(cost_ptFile['value_state_dict'])
        self.cost_function.load_state_dict(cost_ptFile['cost_state_dict'])

    def print_update(self):
        update_message = '[Episode]: {0} | [Avg. Reward]: {1} | [Avg. Cost]: {2}'
        format_args = (self.episode_num, self.mean_rewards[-1], self.mean_costs[-1])
        print(update_message.format(*format_args))


if __name__ == '__main__':
    # setup the environment
    shop_data = pd.read_csv("shopInitialization.csv")  # 从30天数据中提取出来的相关商家（包括经纬度、所在路网网格ID）
    mapped_matrix_int = np.arange(0, 100, 1).reshape([10, 10])  # 用一个矩形来网格化配送地图，不可送达地（湖泊、海洋等）标记-1
    environment = Environment(shop_data, mapped_matrix_int, 10, 10)
    monitor = Monitor(train=True, spec="CMO_RL_Dispatch")

    # get state / action size
    state_size = environment.n_valid_nodes * 3 + 4
    action_size = 7 + 40  # 起点、终点(经度, 纬度), cost, 距离, 同节点订单数目, route
    batch = int(5e+2)  # 5*10^2=500, 每次取出500条经验来更新网络
    capacity = int(1e+4)  # 1.0*10^4=10000, 经验池的capacity

    # config
    max_kl = 1e-2  # 最大KL距离约束（新旧策略更新的KL距离差距）同TRPO算法 0.01
    max_constraint_val = batch * 0.05  # 超时率5%
    val_lr = 1e-2  # Adm优化器学习率 Critic网络
    cost_lr = 1e-2
    val_small_loss = 1e-3  # 在进行参数更新时,学习速率要除以这个积累量的平方根,其中加上一个很小值是为了防止除0的出现
    cost_small_loss = 1e-3

    discount_val = 0.995  # gamma for discounted reward
    discount_cost = 0.995  # gamma for discounted cost
    lambda_val = 0.98  # GAE中：当λ=0时, GAE的形式就是TD误差的形式, 有偏差, 但方差小
    lambda_cost = 0.98  # λ=1时就是蒙特卡洛的形式, 无偏差, 但是方差大

    line_search_max_iter = 10  # 线搜索的次数
    line_search_coefficient = 0.9  # 步长的衰减率（每次搜索时）
    line_search_accept_ratio = 0.1  # 用于loss下降判断 ratio=actual_improve/expected_improve

    cpo = CPO(state_dim=state_size, action_dim=action_size, simulator=environment, capacity=capacity, batch_size=batch,
              max_kl=max_kl, val_lr=val_lr, cost_lr=cost_lr, max_constraint_val=max_constraint_val,
              val_small_loss=1e-3, cost_small_loss=1e-3, Monitor=monitor,
              discount_val=discount_val, discount_cost=discount_cost,
              lambda_val=lambda_val, lambda_cost=lambda_cost,
              line_search_max_iter=line_search_max_iter,
              line_search_coefficient=line_search_coefficient,
              line_search_accept_ratio=line_search_accept_ratio)

    # for train
    n_episodes = 565
    n_steps = 168  # 一天内
    alpha = 0.3  # reward中骑手公平的权重
    beta = 0.3  # reward中商家公平的权重

    cpo.train(n_episodes, n_steps, alpha, beta)
