"""
owner: Zou ying Cao
data: 2023-02-15
description: main function for training
"""
import pandas as pd
from CPO import CPO
from envs.envs import Environment
from envs.order_dispatching import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")

if __name__ == '__main__':
    # setup the environment
    shop_data = pd.read_csv("shopInitialization.csv")  # 从30天数据中提取出来的相关商家（包括经纬度、所在路网网格ID）
    mapped_matrix_int = np.arange(0, 100, 1).reshape([10, 10])  # 用一个矩形来网格化配送地图，不可送达地（湖泊、海洋等）标记-1
    environment = Environment(shop_data, mapped_matrix_int, 10, 10)

    # get state / action size
    state_size = environment.n_valid_nodes * 3 + 4
    action_size = 7 + 40  # 起点、终点(经度, 纬度), cost, 距离, 同节点订单数目, route
    batch = int(5e+2)  # 5*10^2=500, 每次取出500条经验来更新网络
    capacity = int(1e+6)  # 1.0*10^6=1000000, 经验池的capacity

    # config
    max_kl = 1e-2  # 最大KL距离约束（新旧策略更新的KL距离差距）同TRPO算法
    max_constraint_val = batch*0.05  # 超时率5%
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
              device=device, max_kl=max_kl, val_lr=val_lr, cost_lr=cost_lr, max_constraint_val=max_constraint_val,
              val_small_loss=1e-3, cost_small_loss=1e-3,
              discount_val=discount_val, discount_cost=discount_cost,
              lambda_val=lambda_val, lambda_cost=lambda_cost,
              line_search_max_iter=line_search_max_iter,
              line_search_coefficient=line_search_coefficient,
              line_search_accept_ratio=line_search_accept_ratio)

    # for train
    n_episodes = 30
    n_steps = 168  # 一天内
    alpha = 0.3  # reward中骑手公平的权重
    beta = 0.3  # reward中商家公平的权重

    cpo.train(n_episodes, n_steps, alpha, beta)
