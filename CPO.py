"""
owner: Zou ying Cao
data: 2023-02-15
description: Constrained Policy Optimization
"""
import gc
import sys
import random
import pandas as pd
from torch.nn import MSELoss
from models import Actor, ValueCritic, CostCritic

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

        self.batch_size = batch_size
        self.memory_size = memory_size
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

        elif self.curr_lens + len(s) <= self.memory_size:  # len(s)=s.shape[0]
            self.states.extend(s)
            self.state_inputs.extend(sa)
            self.next_states.extend(next_s)
            self.actions.extend(a)
            self.rewards.extend(r)
            self.curr_lens = len(self.states)
            self.cost.extend(c)
        else:
            # popLength = list(range(len(s)))
            for i in range(len(s)):
                self.states.pop(i)
                self.state_inputs.pop(i)
                self.actions.pop(i)
                self.rewards.pop(i)
                self.next_states.pop(i)
                self.cost.pop(i)

            self.states.extend(s)
            self.state_inputs.extend(sa)
            self.next_states.extend(next_s)
            self.actions.extend(a)
            self.rewards.extend(r)
            self.cost.extend(c)

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.state_inputs, self.states, self.rewards, self.next_states, self.cost, self.actions]

        rand = random.randint(0, self.curr_lens - self.batch_size)
        batch_s = self.states[rand:rand + self.batch_size]
        batch_sa = self.state_inputs[rand:rand + self.batch_size]
        batch_a = self.actions[rand:rand + self.batch_size]
        batch_r = self.rewards[rand:rand + self.batch_size]
        batch_next_s = self.next_states[rand:rand + self.batch_size]
        batch_c = self.cost[rand:rand + self.batch_size]

        return batch_sa, batch_s, batch_r, batch_next_s, batch_c, batch_a


class CPO:
    def __init__(self, state_dim, action_dim, capacity, batch_size, simulator, Monitor, max_kl=1e-2,
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

        self.save_every = save_every
        self.print_updates = print_updates
        self.monitor = Monitor

        self.mean_rewards = []
        self.mean_costs = []
        self.mean_value_loss = []
        self.mean_cost_loss = []

        if continue_from_file:
            self.load_session()

        # self.policy.to(device)
        # self.value_function.to(device)
        # self.cost_function.to(device)

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
            del orders_data
            gc.collect()

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
                    state_inputs, states, actions, rewards, next_states, costs = process_memory(
                        self.env.state, dispatch_result)
                    self.replay.add(state_inputs, states, actions, rewards, next_states, costs)

                    self.mean_rewards.append(torch.mean(torch.Tensor(trajectory_rewards)))
                    self.mean_costs.append(torch.mean(torch.Tensor(trajectory_costs)))
                    self.monitor.update_reward(self.episode_num * n_step + i_step, self.mean_rewards[-1])
                    self.monitor.update_cost(self.episode_num * n_step + i_step, self.mean_costs[-1])

            self.episode_num += 1
            for _ in range(10):
                batch_state, batch_s, batch_reward, batch_next_s, batch_cost, batch_action = self.replay.sample()

                J_cost = torch.sum(torch.tensor(batch_cost, dtype=torch.float32))
                batch_s = torch.tensor(batch_s, dtype=torch.float32)
                batch_next_s = torch.tensor(batch_next_s, dtype=torch.float32)

                batch_action = torch.LongTensor(batch_action).view(-1, 1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).view(-1, 1)
                batch_cost = torch.tensor(batch_cost, dtype=torch.float32).view(-1, 1)

                td_target_value = batch_reward + self.discount_val * self.value_function(batch_next_s)
                td_delta_value = td_target_value - self.value_function(batch_s)
                td_target_cost = batch_cost + self.discount_cost * self.cost_function(batch_next_s)
                td_delta_cost = td_target_cost - self.cost_function(batch_s)

                reward_advantage = compute_adv(self.discount_val, self.lambda_val,
                                               td_delta_value.cpu().detach().numpy())
                cost_advantage = compute_adv(self.discount_cost, self.lambda_cost,
                                             td_delta_cost.cpu().detach().numpy())

                # 线性变换(x-mean)/std:不改变数组分布形状，只是将数据的平均数变为0，标准差变为1
                # 标准化后求梯度更快点
                reward_advantage -= reward_advantage.mean()
                reward_advantage /= reward_advantage.std()
                cost_advantage -= cost_advantage.mean()
                cost_advantage /= cost_advantage.std()

                self.update_Actor(batch_state, batch_action, reward_advantage, cost_advantage, J_cost)
                self.update_Critic(self.value_function, self.value_optimizer, batch_s, batch_reward,
                                   self.val_small_loss, trajectory_value_loss)
                self.update_Critic(self.cost_function, self.cost_optimizer, batch_s, batch_cost,
                                   self.cost_small_loss, trajectory_cost_loss)

            self.mean_value_loss.append(torch.mean(torch.Tensor(trajectory_value_loss)))
            self.mean_cost_loss.append(torch.mean(torch.Tensor(trajectory_cost_loss)))
            self.monitor.record_loss(self.episode_num, self.mean_value_loss[-1], False)  # true/false:cost/value
            self.monitor.record_loss(self.episode_num, self.mean_cost_loss[-1], True)

            self.env.update_env()

            if self.print_updates:
                self.print_update()
            if self.save_every and not self.episode_num % self.save_every:
                self.save_session()

    def update_Actor(self, states, actions, reward_advantage, constraint_advantage, J_c):
        self.policy.train()

        log_action_prob = torch.tensor([])
        action_dists = torch.tensor([])
        for i in range(len(states)):
            s = torch.tensor(states[i], dtype=torch.float32)
            action_dist = self.policy(s)
            lg = torch.log(action_dist.gather(0, actions[i]))
            log_action_prob = torch.cat((log_action_prob, lg), 0)
            action_dists = torch.cat((action_dists, action_dist), 0)

        # log_action_prob = torch.log(self.policy(states).gather(1, actions))
        imp_sampling = torch.exp(log_action_prob - log_action_prob.detach()).view(-1, 1)  # 重要性采样

        reward_loss = -torch.mean(imp_sampling * reward_advantage)  # 以平均作为期望,-surrogate_objective
        # print(reward_loss)
        reward_grad = flat_grad(reward_loss, self.policy.parameters(), retain_graph=True)  # 计算梯度

        constraint_loss = torch.mean(imp_sampling * constraint_advantage)
        constraint_grad = flat_grad(constraint_loss, self.policy.parameters(), retain_graph=True)  # 计算梯度

        mean_kl = mean_kl_first_fixed(action_dists.detach(), action_dists)  # KL散度的平均值
        Fvp_fun = get_Hvp_fun(mean_kl, self.policy.parameters())  # 返回的是一个函数：用于计算黑塞矩阵和一个向量的乘积

        # 用共轭梯度法计算x = H^(-1)g
        F_inv_g = cg_solver(Fvp_fun, reward_grad)  # F_inv_g = H^(-1)g, g是目标函数的梯度
        F_inv_b = cg_solver(Fvp_fun, constraint_grad)  # F_inv_b = H^(-1)B, bi是第i个约束的梯度

        q = torch.matmul(reward_grad, F_inv_g)  # q = g^T*H^(-1)g
        c = (J_c - self.max_constraint_val)  # c = J_c(π_k) − d

        EPS = 1e-8
        if torch.matmul(constraint_grad, constraint_grad).item() <= EPS and c < 0:
            # feasible and cost grad is zero --- shortcut to pure TRPO update
            is_feasible = True
            lam = torch.sqrt(q / (2 * self.max_kl))
            nu = 0.0
            search_dir = -(1 / (lam + EPS)) * (F_inv_g + nu * F_inv_b)  # 1/lam*(H^(-1)g-H^(-1)B*nu)
        else:
            r = torch.matmul(reward_grad, F_inv_b)  # r = g^T*(H^(-1)B)
            s = torch.matmul(constraint_grad, F_inv_b)  # s = B^T*(H^(-1)B)x
            # infeasible(即CPO会take a bad step, 计算出来的policy是不满足cost约束的)
            # 只有部分的trust region都在constraint-satisfying half_space内,
            # 此时CMDP的可行解不一定在trust region内, 所以可能需要recovery
            is_feasible = False if c > 0 and c ** 2 / s - 2 * self.max_kl > 0 else True  # delta=1/2*(J_c-max_cost)

            if is_feasible:  # 对偶解法
                lam, nu = self.calc_dual_vars(q, r, s, c)  # dual_vars: 对偶问题最优解(整数规划一般不考虑对偶问题的最优解)
                # print('lam, nu', lam, nu)
                search_dir = -(1 / (lam + EPS)) * (F_inv_g + nu * F_inv_b)  # 1/lam*(H^(-1)g-H^(-1)B*nu)
            else:
                search_dir = -torch.sqrt(
                    2 * self.max_kl / (s + EPS)) * F_inv_b  # sqrt(2*delta/(b^T*H^(-1)*b))*(H^(-1)*b)

        # Should be positive
        expected_loss_improve = torch.matmul(reward_grad, search_dir)
        current_policy = get_flat_params(self.policy)

        def line_search_criterion(search_direction, step_length):  # 判断loss下降，cost在范围内，KL散度在范围内
            # search_direction:下降方向
            # step_length:步长
            test_policy = current_policy + step_length * search_direction
            set_params(self.policy, test_policy)

            with torch.no_grad():  # Test if conditions are satisfied
                test_prob = torch.tensor([])
                test_dists = torch.tensor([])
                for index in range(len(states)):
                    s_ = torch.tensor(states[index], dtype=torch.float32)
                    test_dist = self.policy(s_)
                    lg_ = torch.log(test_dist.gather(0, actions[index]))
                    test_prob = torch.cat((test_prob, lg_), 0)
                    test_dists = torch.cat((test_dists, test_dist), 0)

                importance_sampling = torch.exp(test_prob - log_action_prob.detach())
                test_loss = -torch.mean(importance_sampling * reward_advantage)

                # 判断loss是否下降
                actual_improve = test_loss - reward_loss
                expected_improve = step_length * expected_loss_improve
                loss_cond = actual_improve / expected_improve >= self.line_search_accept_ratio

                # 判断cost是否在范围内
                cost_cond = step_length * torch.matmul(constraint_grad, search_direction) <= max(-c, 0.0)
                # print('test_dists', test_dists)
                # 判断kl散度是否在范围内
                test_kl = mean_kl_first_fixed(action_dists.detach(), test_dists)  # 新旧策略之间的KL距离
                kl_cond = (test_kl <= self.max_kl)

            set_params(self.policy, current_policy)

            if is_feasible:
                return loss_cond and cost_cond and kl_cond

            return cost_cond and kl_cond

        # print('search_dir', search_dir)
        # 若十步内找不到合适的参数, step_len就返回0, 即参数不变
        step_len = line_search(search_dir, 1.0, line_search_criterion,
                               self.line_search_coefficient, self.line_search_max_iter)
        new_policy = current_policy + step_len * search_dir
        set_params(self.policy, new_policy)

    def update_Critic(self, critic, optimizer, states, targets, small_value, loss_record):
        critic.train()

        def mse():
            optimizer.zero_grad()

            predictions = critic(states).view(-1, 1)
            loss = self.mse_loss(predictions, targets)

            flat_params = get_flat_params(critic)
            small_loss = small_value * torch.sum(torch.pow(flat_params, 2))
            loss += small_loss
            loss_record.append(loss)
            loss.backward()

            return loss

        optimizer.step(mse)

    def calc_dual_vars(self, q, r, s, c):
        EPS = 1e-6
        A = q - r ** 2 / (s + EPS)  # should always be positive
        B = 2 * self.max_kl - c ** 2 / (s + EPS)

        if c < 0 and B < 0:  # 信赖域内所有点均有可行解, TRPO的解就是满足CMDP的可行解(feasible)
            lam = torch.sqrt(q / (2 * self.max_kl))
            nu = 0.0
            return lam, nu

        # B>=0的情况
        lam_mid = r / (c + EPS)
        lam_a = torch.sqrt(A / B)
        lam_b = torch.sqrt(q / (2 * self.max_kl))

        # f_a公式为：-0.5*(A/lam+B*lam)-r*c/s
        # f_b公式为：-0.5*(q/lam+2*kl*lam)
        f_mid = -0.5 * (q / (lam_mid + EPS) + lam_mid * 2 * self.max_kl)
        f_a = -torch.sqrt(A * B) - r * c / (s + EPS)  # lam_a=sqrt(A/B)代入f_a的结果
        f_b = -torch.sqrt(2 * q * self.max_kl)  # lam_b=sqrt(q/(2*kl))代入f_b的结果

        if lam_mid > 0:
            if c < 0:
                # lam_a = max(0,min(r/c,sqrt(A/B)))
                if lam_a > lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid  # lam_a=r/c代入f_a的结果
                # lam_b = max(r/c,sqrt(q/(2*kl)))
                if lam_b < lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid  # lam_b=r/c代入f_b的结果
            else:
                if lam_a < lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid
                if lam_b > lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid
            lam = lam_a if f_a >= f_b else lam_b
        else:
            # lam = lam_a if f_a >= f_b else lam_b
            if c < 0:  # lam_a = 0,lam_b = sqrt(q/(2*kl)), f_a = 无穷小 < f_b
                lam = lam_b
            else:  # lam_a = sqrt(A/B),lam_b = 0, f_a > f_b = 无穷小
                lam = lam_a

        nu = max(0.0, (lam * c - r) / (s + EPS))

        return lam, nu

    def save_session(self):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, 'CMO_RL_Dispatch' + '.pt')

        ptFile = dict(policy_state_dict=self.policy.state_dict(),
                      value_state_dict=self.value_function.state_dict(),
                      cost_state_dict=self.cost_function.state_dict(),
                      mean_rewards=self.mean_rewards,
                      mean_costs=self.mean_costs,
                      mean_value_loss=self.mean_value_loss,
                      mean_cost_loss=self.mean_cost_loss,
                      episode_num=self.episode_num)

        torch.save(ptFile, save_path)

    def load_session(self):
        load_path = os.path.join(save_dir, 'model0329' + '.pt')
        ptFile = torch.load(load_path)

        self.policy.load_state_dict(ptFile['policy_state_dict'])
        # self.value_function.load_state_dict(ptFile['value_state_dict'])
        # self.cost_function.load_state_dict(ptFile['cost_state_dict'])
        # self.mean_rewards = ptFile['mean_rewards']
        # self.mean_costs = ptFile['mean_costs']
        # self.mean_value_loss = ptFile['mean_value_loss']
        # self.mean_cost_loss = ptFile['mean_cost_loss']
        # self.episode_num = ptFile['episode_num']

    def print_update(self):
        update_message = '[Episode]: {0} | [Avg. Reward]: {1} | [Avg. Cost]: {2}'
        format_args = (self.episode_num, self.mean_rewards[-1], self.mean_costs[-1])
        print(update_message.format(*format_args))
