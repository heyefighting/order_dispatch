"""
owner: Zou ying Cao
data: 2023-02-15
description: Actor Critic
"""
import torch
from torch.nn import Module
import torch.nn.functional as F


class Actor(Module):
    def __init__(self, state_dim_, action_dim_):
        super(Actor, self).__init__()
        self.state_dim = state_dim_
        self.action_dim = action_dim_
        self.S = torch.nn.Linear(state_dim_, 128)
        self.A = torch.nn.Linear(action_dim_, 8)
        self.l1 = torch.nn.Linear(128 + 8, 16)
        self.l2 = torch.nn.Linear(16, 8)
        self.f = torch.nn.Linear(8, 1)

    def forward(self, X):
        s1 = X[:, :self.state_dim]
        a1 = X[:, -self.action_dim:]
        S1 = F.relu(self.S(s1))
        A1 = F.relu(self.A(a1))
        Y1 = torch.cat((S1, A1), dim=1)
        l1 = F.relu(self.l1(Y1))
        l2 = F.relu(self.l2(l1))
        return torch.softmax(F.relu(self.f(l2)) + 1, dim=0).squeeze(1)  # 从shape(x,1)二维变为shape(x)一维


class Critic(Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.S = torch.nn.Linear(state_dim, 128)
        self.l1 = torch.nn.Linear(128, 64)
        self.l2 = torch.nn.Linear(64, 32)
        self.f = torch.nn.Linear(32, 1)

    def forward(self, X):
        S1 = F.relu(self.S(X))
        l1 = F.relu(self.l1(S1))
        l2 = F.relu(self.l2(l1))
        return F.relu(self.f(l2))
