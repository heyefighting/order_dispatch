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
        if self.train:
            self.value_loss_window = self.vis.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros(1).cpu(),  # torch.zeros((1))
                opts=dict(xlabel='episode',
                          ylabel='mle loss for value Critic Network',
                          title='Training Loss' + spec,
                          legend=['Loss']))
            self.cost_loss_window = self.vis.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros(1).cpu(),  # torch.zeros((1))
                opts=dict(xlabel='episode',
                          ylabel='mle loss for cost Critic Network',
                          title='Training Loss' + spec,
                          legend=['Loss']))
        self.log_file = None
        self.value_window = None
        self.cost_window = None
        self.text_window = None

    def update(self, step, reward, cost):
        if self.value_window is None:
            self.value_window = self.vis.line(X=torch.Tensor([step]).cpu(),
                                              Y=torch.Tensor([reward]).cpu(),
                                              opts=dict(xlabel='step',
                                                        ylabel='reward',
                                                        title='Value Dynamics' + self.spec,
                                                        legend=['reward']))
            self.cost_window = self.vis.line(X=torch.Tensor([step]).cpu(),
                                             Y=torch.Tensor([cost]).cpu(),
                                             opts=dict(xlabel='step',
                                                       ylabel='cost',
                                                       title='Cost Dynamics' + self.spec,
                                                       legend=['cost']))
        else:
            self.vis.line(
                X=torch.Tensor([step]).cpu(),
                Y=torch.Tensor([reward]).cpu(),
                win=self.value_window,
                update='append')
            self.vis.line(
                X=torch.Tensor([step]).cpu(),
                Y=torch.Tensor([cost]).cpu(),
                win=self.cost_window,
                update='append')

    def record_loss(self, step, loss=None, flag=False):
        if self.train:
            if flag:
                window = self.cost_loss_window
            else:
                window = self.value_loss_window
            self.vis.line(
                X=torch.Tensor([step]).cpu(),
                Y=torch.Tensor([loss]).cpu(),
                win=window,
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
