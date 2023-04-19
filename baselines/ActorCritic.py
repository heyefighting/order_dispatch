"""
owner: Zou ying Cao
data: 2023-04-04
description:
"""
import torch.nn.functional as F
from models import Actor, ValueCritic
from monitor import Monitor
from CPO import ReplayMemory
from envs.envs import Environment
from env_utils.neighbours import *
from envs.order_dispatching import *

save_dir = 'save-dir'


class AC:
    def __init__(self, state_dim, action_dim,gamma, simulator, capacity_size, batch_size, actor_lr, critic_lr, epsilon,
                 Monitor, continue_from_file=False, save_every=1, print_updates=True):
        self.env = simulator
        self.epsilon = epsilon

        self.actor = Actor(state_dim, action_dim)
        self.critic = ValueCritic(state_dim)
        # 使用Adam优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器

        self.replay = ReplayMemory(capacity_size, batch_size)
        self.monitor = Monitor
        self.mean_rewards = []
        self.mean_costs = []
        self.mean_actor_loss = []
        self.mean_critic_loss = []
        self.episode_num = 0
        self.gamma = gamma  # 折扣因子

        self.continue_from_file = continue_from_file
        self.save_every = save_every
        self.print_updates = print_updates

        if self.continue_from_file:
            self.load_session()

    def take_action(self, state):
        state_ = torch.tensor(state, dtype=torch.float32)
        action_prob = self.actor(state_)
        if np.random.random() < self.epsilon:
            c_id = torch.distributions.Categorical(action_prob).sample().item()
        else:
            maxi = max(action_prob)
            c_id = np.random.choice([index for index, t in enumerate(action_prob) if t == maxi])
        return c_id

    def update(self, s, states, actions, rewards, next_s, t_actor_loss, t_critic_loss):
        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_s)
        td_delta = td_target - self.critic(s)  # 时序差分误差
        critic_loss = torch.mean(F.mse_loss(self.critic(s), td_target.detach()))  # 均方误差损失函数
        t_critic_loss.append(critic_loss)
        log_action_prob = torch.tensor([])
        for i in range(len(states)):
            s = torch.tensor(states[i], dtype=torch.float32)
            action_dist = self.actor(s)
            lg = torch.log(action_dist.gather(0, actions[i]))
            log_action_prob = torch.cat((log_action_prob, lg), 0)

        actor_loss = torch.mean(-log_action_prob * td_delta.detach())
        t_actor_loss.append(actor_loss)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

    def train(self, n_episodes, n_step, alpha, beta, penalty_rate):
        trajectory_actor_loss = []
        trajectory_critic_loss = []

        while self.episode_num < n_episodes:
            fileName = "../datasets/orderData" + str(self.episode_num + 1) + ".csv"
            orders_data = pd.read_csv(fileName)
            self.env.reset_env(orders_data)  # 出现新的订单与用户

            trajectory_actor_loss.clear()
            trajectory_critic_loss.clear()
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
                        d.set_penalty_reward(self.env, alpha, beta, penalty_rate, self.episode_num, record)

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
            # if self.episode_num % 2 == 0:
            #     self.epsilon = max(0, self.epsilon - 0.02)
            for _ in range(10):
                batch_state, batch_s, batch_reward, batch_next_s, _, batch_action = self.replay.sample()
                batch_action = torch.LongTensor(batch_action).view(-1, 1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float).view(-1, 1)
                batch_s = torch.tensor(batch_s, dtype=torch.float32)
                batch_next_s = torch.tensor(batch_next_s, dtype=torch.float32)
                self.update(batch_s, batch_state, batch_action, batch_reward, batch_next_s,
                            trajectory_actor_loss, trajectory_critic_loss)

            self.mean_actor_loss.append(torch.mean(torch.Tensor(trajectory_actor_loss)))
            self.mean_critic_loss.append(torch.mean(torch.Tensor(trajectory_critic_loss)))
            self.monitor.record_loss(self.episode_num, self.mean_actor_loss[-1], False)  # true/false:cost/value
            self.monitor.record_loss(self.episode_num, self.mean_critic_loss[-1], True)

            self.env.update_env()

            if self.print_updates:
                self.print_update()
            if self.save_every and not self.episode_num % self.save_every:
                self.save_session()

    def save_session(self):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, 'ac_model0404' + '.pt')

        ptFile = dict(policy_state_dict=self.actor.state_dict(),
                      critic_state_dict=self.critic.state_dict(),
                      mean_rewards=self.mean_rewards,
                      mean_actor_loss=self.mean_actor_loss,
                      mean_critic_loss=self.mean_critic_loss,
                      episode_num=self.episode_num)

        torch.save(ptFile, save_path)

    def load_session(self):
        actor_load_path = os.path.join(save_dir, 'ac_model_1_0405' + '.pt')
        actor_ptFile = torch.load(actor_load_path)

        self.actor.load_state_dict(actor_ptFile['policy_state_dict'])
        self.critic.load_state_dict(actor_ptFile['critic_state_dict'])

    def print_update(self):
        update_message = '[Episode]: {0} | [Avg. Reward]: {1}'
        format_args = (self.episode_num, self.mean_rewards[-1])
        print(update_message.format(*format_args))


if __name__ == '__main__':
    # setup the environment
    shop_data = pd.read_csv("../shopInitialization.csv")  # 从30天数据中提取出来的相关商家（包括经纬度、所在路网网格ID）
    mapped_matrix_int = np.arange(0, 100, 1).reshape([10, 10])  # 用一个矩形来网格化配送地图，不可送达地（湖泊、海洋等）标记-1
    environment = Environment(shop_data, mapped_matrix_int, 10, 10)
    monitor = Monitor(train=True, spec="ActorCritic_Dispatch")

    # get state / action size
    state_size = environment.n_valid_nodes * 3 + 4
    action_size = 7 + 40  # 起点、终点(经度, 纬度), cost, 距离, 同节点订单数目, route
    batch = int(5e+2)  # 5*10^2=500, 每次取出500条经验来更新网络
    capacity = int(1e+4)  # 1.0*10^4=10000, 经验池的capacity

    model = AC(state_dim=state_size, action_dim=action_size, simulator=environment, actor_lr=1e-2, critic_lr=1e-2,
               capacity_size=capacity, batch_size=batch, Monitor=monitor, gamma=0.995, epsilon=0.02)

    # for train
    n_episodes = 565
    n_steps = 168  # 一天内
    alpha = 0.3  # reward中骑手公平的权重
    beta = 0.3  # reward中商家公平的权重
    penalty_rate = 20

    model.train(n_episodes, n_steps, alpha, beta, penalty_rate)
