"""
owner: Zou ying Cao
data: 2023-02-01
description:
"""
import torch
import visdom


class Monitor(object):

    def __init__(self, train=False, spec=''):
        self.vis = visdom.Visdom()
        self.train = train
        self.spec = spec
        self.value_window = self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros(1).cpu(),  # torch.zeros((1))
            opts=dict(xlabel='step',
                      ylabel='reward',
                      title='Value_Dynamics_' + self.spec,
                      legend=['reward']))
        self.cost_window = None
        if self.train:
            self.value_loss_window = self.vis.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros(1).cpu(),  # torch.zeros((1))
                opts=dict(xlabel='episode',
                          ylabel='mle loss for value Critic Network',
                          title='Value_Critic_Loss_' + spec,
                          legend=['Loss']))
            self.cost_loss_window = None
        self.log_file = None
        self.text_window = None

    def update_reward(self, step, reward):
        self.vis.line(
            win=self.value_window,
            X=torch.Tensor([step]).cpu(),
            Y=torch.Tensor([reward]).cpu(),
            update='append')

    def update_cost(self, step, cost):
        if self.cost_window is None:
            self.cost_window = self.vis.line(
                X=torch.Tensor([step]).cpu(),
                Y=torch.Tensor([cost]).cpu(),
                opts=dict(xlabel='step',
                          ylabel='cost',
                          title='Cost_Dynamics_' + self.spec,
                          legend=['cost']))
        else:
            self.vis.line(
                    win=self.cost_window,
                    X=torch.Tensor([step]).cpu(),
                    Y=torch.Tensor([cost]).cpu(),
                    update='append')

    def record_loss(self, step, loss=None, flag=False):
        if self.train:
            if flag:
                if self.cost_loss_window is None:
                    self.cost_loss_window = self.vis.line(
                        X=torch.zeros((1,)).cpu(),
                        Y=torch.zeros(1).cpu(),  # torch.zeros((1))
                        opts=dict(xlabel='episode',
                                  ylabel='mle loss for cost Critic Network',
                                  title='Cost_Critic_Loss_' + self.spec,
                                  legend=['Loss']))
                self.vis.line(
                    win=self.cost_loss_window,
                    X=torch.Tensor([step]).cpu(),
                    Y=torch.Tensor([loss]).cpu(),
                    update='append')
            else:
                self.vis.line(
                    win=self.value_loss_window,
                    X=torch.Tensor([step]).cpu(),
                    Y=torch.Tensor([loss]).cpu(),
                    update='append')

    def text(self, tt):
        if self.text_window is None:
            self.text_window = self.vis.text("QPath" + self.spec)
        self.vis.text(
            tt,
            win=self.text_window,
            append=True)

    def init_log(self, save_path, name):
        self.log_file = open("{}{}.log".format(save_path, name), 'w')

    def add_log(self, state, action, reward, terminal, preference):
        self.log_file.write("{}\t{}\t{}\t{}\t{}\n".format(state, action, reward, terminal, preference.cpu().numpy()))
