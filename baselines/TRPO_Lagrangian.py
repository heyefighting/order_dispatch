"""
owner: Zou ying Cao
data: 2023-04-06
description: TRPO + 拉格朗日方法
"""
from torch.nn import MSELoss

from envs.envs import Environment
from env_utils.neighbours import *
from envs.order_dispatching import *
from models import Actor, ValueCritic, CostCritic
from monitor import Monitor
from CPO import ReplayMemory, compute_adv
from baselines.lagrangian_base import Lagrangian
from optimization_utils.hvp import get_Hvp_fun
from optimization_utils.line_search import line_search
from optimization_utils.conjugate_gradient import cg_solver
from torch_utils.distribution_utils import mean_kl_first_fixed
from torch_utils.torch_utils import flat_grad, get_flat_params, set_params

save_dir = 'save-dir'


class TRPO_Lag(Lagrangian):
    def __init__(
            self, state_dim, action_dim, capacity, batch_size, simulator, Monitor, max_kl, epochs, entropy_coef,
            epsilon, val_small_loss, cost_small_loss, discount_val, discount_cost, lambda_val, lambda_cost, ac_lr,
            val_lr, cost_limit=25., clip=0.2,
            lagrangian_multiplier_init=10.0, lambda_lr=0.035, lambda_optimizer='Adam',
            line_search_coefficient=0.9, line_search_max_iter=10, line_search_accept_ratio=0.1,
            continue_from_file=True, save_every=1, print_updates=True
    ):
        Lagrangian.__init__(
            self,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer
        )

        self.clip = clip
        self.max_kl = max_kl
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
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
        self.monitor = Monitor
        self.mse_loss = MSELoss(reduction='mean')
        self.actor_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=ac_lr)
        self.value_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=val_lr)
        self.cost_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=val_lr)

        # configs
        self.epochs = epochs
        self.episode_num = 12
        self.batch_size = batch_size
        self.val_small_loss = val_small_loss
        self.cost_small_loss = cost_small_loss
        self.discount_val = discount_val
        self.discount_cost = discount_cost
        self.lambda_val = lambda_val
        self.lambda_cost = lambda_cost

        self.line_search_coefficient = line_search_coefficient
        self.line_search_max_iter = line_search_max_iter
        self.line_search_accept_ratio = line_search_accept_ratio

        self.mean_rewards = []
        self.mean_costs = []
        self.mean_ac_loss = []
        self.mean_value_loss = []
        self.mean_cost_loss = []

        self.continue_from_file = continue_from_file
        self.save_every = save_every
        self.print_updates = print_updates

        if self.continue_from_file:
            self.load_session()

    def update(self, trajectory_value_loss, trajectory_cost_loss):
        batch_state, batch_s, batch_reward, batch_next_s, batch_cost, batch_action = self.replay.sample()

        ep_costs = torch.sum(torch.tensor(batch_cost, dtype=torch.float32))
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

        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)

        # now update policy and value network
        for _ in range(10):
            self.update_Actor(batch_state, batch_action, reward_advantage, cost_advantage)
            self.update_Critic(self.value_function, self.value_optimizer, batch_s, batch_reward,
                               self.val_small_loss, trajectory_value_loss)
            self.update_Critic(self.cost_function, self.cost_optimizer, batch_s, batch_cost,
                               self.cost_small_loss, trajectory_cost_loss)

    def update_Actor(self, states, actions, reward_advantage, cost_advantage):
        self.policy.train()

        log_probs = torch.tensor([])
        action_dists = torch.tensor([])
        action_dists_entropy = torch.tensor([])
        for i in range(len(states)):
            s = torch.tensor(states[i], dtype=torch.float32)
            action_dist = self.policy(s)
            lg = torch.log(action_dist.gather(0, actions[i]))
            log_probs = torch.cat((log_probs, lg), 0)
            action_dists = torch.cat((action_dists, action_dist), 0)
            action_dist_entropy = torch.distributions.Categorical(action_dist).entropy()  # .entropy().mean()
            action_dists_entropy = torch.cat((action_dists_entropy, torch.tensor([action_dist_entropy])), 0)

        ratio = torch.exp(log_probs - log_probs.detach())
        actor_loss = -torch.mean(ratio * reward_advantage)  # PPO损失函数
        actor_loss -= self.entropy_coef * action_dists_entropy.mean()  # .entropy().mean()

        # ensure that lagrange multiplier is positive
        penalty = torch.clamp_min(self.lagrangian_multiplier, 0.0)
        actor_loss += penalty * ((ratio * cost_advantage).mean())
        actor_loss /= (1 + penalty)
        reward_grad = flat_grad(actor_loss, self.policy.parameters(), retain_graph=True)  # 计算梯度

        mean_kl = mean_kl_first_fixed(action_dists.detach(), action_dists)  # KL散度的平均值
        Fvp_fun = get_Hvp_fun(mean_kl, self.policy.parameters())  # 返回的是一个函数：用于计算黑塞矩阵和一个向量的乘积

        # 用共轭梯度法计算x = H^(-1)g
        F_inv_g = cg_solver(Fvp_fun, reward_grad)  # F_inv_g = H^(-1)g, g是目标函数的梯度
        q = torch.matmul(reward_grad, F_inv_g)  # q = g^T*H^(-1)g
        search_dir = -torch.sqrt(2 * self.max_kl / (q + 1e-8)) * F_inv_g  # 有没有负号啊

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

                importance_sampling = torch.exp(test_prob - log_probs.detach())
                test_loss = -torch.mean(importance_sampling * reward_advantage)

                # 判断loss是否下降
                actual_improve = test_loss - actor_loss
                expected_improve = step_length * expected_loss_improve
                loss_cond = actual_improve / expected_improve >= self.line_search_accept_ratio

                # 判断kl散度是否在范围内
                test_kl = mean_kl_first_fixed(action_dists.detach(), test_dists)  # 新旧策略之间的KL距离
                kl_cond = (test_kl <= self.max_kl)

            set_params(self.policy, current_policy)
            return loss_cond and kl_cond

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

    def take_action(self, state):
        state_ = torch.tensor(state, dtype=torch.float32)
        action_prob = self.policy(state_)
        if np.random.random() < self.epsilon:
            c_id = torch.distributions.Categorical(action_prob).sample().item()
        else:
            maxi = max(action_prob)
            c_id = np.random.choice([index for index, t in enumerate(action_prob) if t == maxi])
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
                        c_id = self.take_action(state_input)  # , softmax_V

                        i_order.set_order_accept_time(self.env.time_slot_index)  # 订单设置接单时间
                        self.env.shops_dict[i_order.shop_loc].add_order(i_order)  # 相应商家order_list添加该订单
                        # courier部分状态更新包含num_order与接待订单的时间点属性
                        self.env.couriers_dict[id_list[c_id]].take_order(i_order, self.env)
                        self.env.nodes[i_order.begin_p].order_num -= 1

                        d = DispatchPair(i_order, self.env.couriers_dict[id_list[c_id]])
                        d.set_state_input(state_input)  # [c_id]
                        d.set_state(state_couriers[c_id])
                        d.set_action(c_id)  # int
                        d.set_cost(wait_time[c_id])  # 0:不超时, 1:超时

                        record = False
                        if i_step == n_step and self.env.time_slot_index == 167:
                            record = True
                        d.set_reward(self.env, alpha, beta, self.episode_num, record)

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
                    # state_inputs, states, actions, rewards, next_states, costs = process_memory(
                    #     self.env.state, dispatch_result)
                    # self.replay.add(state_inputs, states, actions, rewards, next_states, costs)

                    self.mean_rewards.append(torch.mean(torch.Tensor(trajectory_rewards)))
                    self.mean_costs.append(torch.mean(torch.Tensor(trajectory_costs)))
                    self.monitor.update_reward(self.episode_num * n_step + i_step, self.mean_rewards[-1])
                    self.monitor.update_cost(self.episode_num * n_step + i_step, self.mean_costs[-1])

            self.episode_num += 1
            # self.update(trajectory_value_loss, trajectory_cost_loss)
            #
            # self.mean_value_loss.append(torch.mean(torch.Tensor(trajectory_value_loss)))
            # self.mean_cost_loss.append(torch.mean(torch.Tensor(trajectory_cost_loss)))
            # self.monitor.record_loss(self.episode_num, self.mean_value_loss[-1], False)  # true/false:cost/value
            # self.monitor.record_loss(self.episode_num, self.mean_cost_loss[-1], True)

            self.env.update_env()

            if self.print_updates:
                self.print_update()
            # if self.save_every and not self.episode_num % self.save_every:
            #     self.save_session()

    def save_session(self):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, 'trpo_lag_model0415' + '.pt')

        ptFile = dict(policy_state_dict=self.policy.state_dict(),
                      valueCritic_state_dict=self.value_function.state_dict(),
                      costCritic_state_dict=self.cost_function.state_dict(),
                      mean_rewards=self.mean_rewards,
                      mean_value_loss=self.mean_value_loss,
                      mean_cost_loss=self.mean_cost_loss,
                      episode_num=self.episode_num)

        torch.save(ptFile, save_path)

    def load_session(self):
        actor_load_path = os.path.join(save_dir, 'trpo_lag_model_0413' + '.pt')  # 效果：2>0>1
        actor_ptFile = torch.load(actor_load_path)

        self.policy.load_state_dict(actor_ptFile['policy_state_dict'])
        self.cost_function.load_state_dict(actor_ptFile['costCritic_state_dict'])
        self.value_function.load_state_dict(actor_ptFile['valueCritic_state_dict'])

    def print_update(self):
        update_message = '[Episode]: {0} | [Avg. Reward]: {1}| [Avg. Cost]: {2}'
        format_args = (self.episode_num, self.mean_rewards[-1], self.mean_costs[-1])
        print(update_message.format(*format_args))


if __name__ == '__main__':
    # setup the environment
    shop_data = pd.read_csv("../shopInitialization.csv")  # 从30天数据中提取出来的相关商家（包括经纬度、所在路网网格ID）
    mapped_matrix_int = np.arange(0, 100, 1).reshape([10, 10])  # 用一个矩形来网格化配送地图，不可送达地（湖泊、海洋等）标记-1
    environment = Environment(shop_data, mapped_matrix_int, 10, 10)
    monitor = Monitor(train=True, spec="TRPO_Lag_RL_Dispatch")

    # get state / action size
    state_size = environment.n_valid_nodes * 3 + 4
    action_size = 7 + 40  # 起点、终点(经度, 纬度), cost, 距离, 同节点订单数目, route
    batch = int(5e+2)  # 5*10^2=500, 每次取出500条经验来更新网络
    capacity = int(1e+4)  # 1.0*10^4=10000, 经验池的capacity

    # config
    max_kl = 1e-2  # 最大KL距离约束（新旧策略更新的KL距离差距）同TRPO算法 0.01
    max_constraint_val = batch * 0.05  # 超时率5%
    val_lr = 1e-2  # Adm优化器学习率 Critic网络
    ac_lr = 1e-2
    val_small_loss = 1e-3  # 在进行参数更新时,学习速率要除以这个积累量的平方根,其中加上一个很小值是为了防止除0的出现
    cost_small_loss = 1e-3

    discount_val = 0.995  # gamma for discounted reward
    discount_cost = 0.995  # gamma for discounted cost
    lambda_val = 0.98  # GAE中：当λ=0时, GAE的形式就是TD误差的形式, 有偏差, 但方差小
    lambda_cost = 0.98  # λ=1时就是蒙特卡洛的形式, 无偏差, 但是方差大

    model = TRPO_Lag(state_dim=state_size, action_dim=action_size, simulator=environment, capacity=capacity,
                     batch_size=batch, cost_limit=max_constraint_val, max_kl=max_kl, epochs=10, entropy_coef=0.01,
                     val_small_loss=1e-3, cost_small_loss=1e-3, Monitor=monitor, ac_lr=1e-2, val_lr=1e-2,
                     discount_val=discount_val, discount_cost=discount_cost, epsilon=0.02,
                     lambda_val=lambda_val, lambda_cost=lambda_cost
                     )

    # for train
    n_episodes = 20
    n_steps = 168  # 一天内
    alpha = 0.3  # reward中骑手公平的权重
    beta = 0.3  # reward中商家公平的权重

    model.train(n_episodes, n_steps, alpha, beta)
