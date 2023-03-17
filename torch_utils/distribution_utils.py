"""
owner: Zou ying Cao
data: 2023-02-13
description:
"""
import torch
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


def detach_dist(dist):
    """
    Return a copy of dist with the distribution parameters detached from the
    computational graph
    Parameters
    ----------
    dist: torch.distributions.distribution.Distribution
        the distribution object for which the detached copy is to be returned
    Returns
    -------
    detached_dist
        the detached distribution
    """

    if type(dist) is Categorical:
        detached_dist = Categorical(logits=dist.logits.detach())
    elif type(dist) is Independent:
        detached_dist = Normal(loc=dist.mean.detach(), scale=dist.stddev.detach())
        detached_dist = Independent(detached_dist, 1)

    return detached_dist


def mean_kl_first_fixed(dist_1_detached, dist_2):
    """
    Calculate the kl-divergence between dist_1 and dist_2 after detaching dist_1
    from the computational graph
    Parameters
    ----------
    dist_1_detached : 真实概率分布 p
        the first argument to the kl-divergence function (will be fixed)
    dist_2 : 不真实的概率分布 q
        the second argument to the kl-divergence function (will not be fixed)
    Returns
    -------
    mean_kl :
        the kl-divergence between dist_1 and dist_2
    """
    kl = torch.sum(
        dist_1_detached * (F.log_softmax(dist_1_detached, dim=-1) - F.log_softmax(torch.tensor(dist_2), dim=-1)), 1)
    mean_kl = torch.mean(kl)

    return mean_kl
