"""
owner: Zou ying Cao
data: 2023-02-13
description: 计算海森矩阵和向量的乘积（hessian_matrix_vector_product）
"""
import torch
from torch_utils.torch_utils import flat_grad


def get_Hvp_fun(functional_output, inputs, damping_coefficient=0.0):
    """
    Returns a function that calculates a Hessian-vector product with the Hessian
    of functional_output w.r.t. inputs
    Parameters
    ----------
    functional_output :  (with requires_grad=True)
        the output of the function of which the Hessian is calculated
    inputs :
        the inputs w.r.t. which the Hessian is calculated
    damping_coefficient :
        the multiple of the identity matrix to be added to the Hessian
    """

    inputs = list(inputs)
    # 平均KL散度(output)关于policy参数(input)求梯度
    grad_f = flat_grad(functional_output, inputs, create_graph=True)

    # 先用梯度和向量点乘后计算梯度: Hv =(平均KL散度的梯度*v)再求梯度
    def Hvp_fun(v, retain_graph=True):
        g_v = torch.matmul(grad_f, v)  # 点乘：平均KL散度的梯度*v
        Hvp = flat_grad(g_v, inputs, retain_graph=retain_graph)  # 再求梯度
        Hvp += damping_coefficient * v

        return Hvp

    return Hvp_fun
