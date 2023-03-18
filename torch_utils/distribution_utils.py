"""
owner: Zou ying Cao
data: 2023-02-13
description:
"""
import torch
from torch.distributions.kl import kl_divergence


def mean_kl_first_fixed(dist_1_detached, dist_2):
    """
    Calculate the kl-divergence between dist_1 and dist_2 after detaching dist_1
    from the computational graph
    Parameters
    ----------
    dist_1_detached : 真实概率分布 p detached
        the first argument to the kl-divergence function (will be fixed)
    dist_2 : 不真实的概率分布 q
        the second argument to the kl-divergence function (will not be fixed)
    Returns
    -------
    mean_kl :
        the kl-divergence between dist_1 and dist_2
    """
    # kl = torch.sum(
    #    dist_1_detached * (F.log_softmax(dist_1_detached, dim=-1) - F.log_softmax(dist_2, dim=-1)), 1)
    kl = kl_divergence(
        torch.distributions.Categorical(dist_1_detached), torch.distributions.Categorical(dist_2))
    mean_kl = torch.mean(kl)

    return mean_kl
