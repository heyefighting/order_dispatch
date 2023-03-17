"""
owner: Zou ying Cao
data: 2023-02-13
description:共轭梯度法计算x=(H^-1)*g，避免计算和存储黑塞矩阵的逆矩阵会耗费大量的内存资源和时间
"""
import torch


def cg_solver(Avp_fun, b, device, max_iter=10):
    """
    Finds an approximate solution to a set of linear equations Ax = b
    Parameters
    ----------
    Avp_fun : callable
        a function that right multiplies a matrix A by a vector
    b :
        the right hand term in the set of linear equations Ax = b
    max_iter : int
        the maximum number of iterations (default is 10)
    device:
        cuda or cpu
    Returns
    -------
    x : torch.FloatTensor
        the approximate solution to the system of equations defined by Avp_fun and b

    """

    device = device
    x = torch.zeros_like(b).to(device)  # x0=0
    r = b.clone()  # r0=g,梯度grad
    p = b.clone()  # p0=r0
    rdotr = torch.matmul(r, r)
    for i in range(max_iter):  # 共轭梯度主循环
        if rdotr < 1e-10:
            return x
        # Hp = H*(p_k)
        Hp = Avp_fun(p, retain_graph=True)  # hessian_matrix_vector_product
        # alpha_k = (r_k)^T*(r_k)/((p_k)^T*Hp)
        alpha = rdotr / torch.matmul(p, Hp)
        # x_k+1 = x_k+alpha_k*p_k
        x += alpha * p
        if i == max_iter - 1:
            return x
        # r_k+1 = r_k-alpha_k*Hp_k
        r -= r - alpha * Hp
        # beta = (r_k+1)^T*(r_k+1)/((r_k)^T*(r_k))
        new_rdotr = torch.matmul(r, r)
        beta = new_rdotr / rdotr
        # p_k+1 = r_k+1+beta*p_k
        p = r + beta * p
        rdotr = new_rdotr
