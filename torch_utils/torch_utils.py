"""
owner: Zou ying Cao
data: 2023-02-13
description:
"""
import torch
from torch.autograd import grad


def get_device():
    """
    Return a torch.device object. Returns a CUDA device if it is available and
    a CPU device otherwise.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


save_dir = 'saved-sessions'


def set_params(parameterized_fun, new_params):
    """
    Set the parameters of parameterized_fun to new_params
    Parameters
    ----------
    parameterized_fun :
        the function approximator to be updated
    new_params :
        a flattened version of the parameters to be set
    """

    n = 0

    for param in parameterized_fun.parameters():
        numel = param.numel()
        new_param = new_params[n:n + numel].view(param.size())
        param.data = new_param
        n += numel


def flatten(tensors):
    """
    Return an unrolled, concatenated copy of vecs
    Parameters
    ----------
    tensors : list
        a list of Pytorch Tensor objects
    Returns
    -------
    flattened : torch.FloatTensor
        the flattened version of vecs
    """

    flattened = torch.cat([v.view(-1) for v in tensors])

    return flattened


def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
    """
    Return a flattened view of the gradients of functional_output w.r.t. inputs
    Parameters
    ----------
    functional_output :
        The output of the function for which the gradient is to be calculated
    inputs : (with requires_grad=True)
        the variables w.r.t. which the gradient will be computed
    retain_graph : bool
        whether to keep the computational graph in memory after computing the
        gradient (not required if create_graph is True)
    create_graph : bool
        whether to create a computational graph of the gradient computation
        itself
    Return
    ------
    flat_grads :
        a flattened view of the gradients of functional_output w.r.t. inputs
    """

    if create_graph:
        retain_graph = True

    grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = flatten(grads)  # 即torch.cat([g.view(-1) for g in grads])

    return flat_grads


def get_flat_params(parameterized_fun):
    """
    Get a flattened view of the parameters of a function approximator
    Parameters
    ----------
    parameterized_fun :
        the function approximator for which the parameters are to be returned
    Returns
    -------
    flat_params :
        a flattened view of the parameters of parameterized_fun
    """
    parameters = parameterized_fun.parameters()
    flat_params = flatten([param.view(-1) for param in parameters])

    return flat_params


def normalize(x):
    mean = torch.mean(x)
    std = torch.std(x)
    x_norm = (x - mean) / std

    return x_norm
