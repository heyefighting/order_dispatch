"""
owner: Zou ying Cao
data: 2023-02-13
description: 由于 TRPO 算法用到了泰勒展开的 1 阶和 2 阶近似，这并非精准求解，
因此，新策略可能未必比旧策略好，或未必能满足 KL 散度限制。
TRPO 在每次迭代的最后进行一次线性搜索（Line Search），以确保找到满足条件
"""
import torch


def line_search(search_dir, max_step_len, constraints_satisfied, line_search_coefficient=0.9,
                max_iter=10):
    """
    Perform a backtracking line search that terminates when constraints_satisfied
    return True and return the calculated step length. Return 0.0 if no step
    length can be found for which constraints_satisfied returns True
    Parameters
    ----------
    search_dir :
        the search direction along which the line search is done
    max_step_len :
        the maximum step length to consider in the line search
    constraints_satisfied : callable
        a function that returns a boolean indicating whether the constraints
        are met by the current step length
    line_search_coefficient :
        the proportion by which to reduce the step length after each iteration
    max_iter :
        the maximum number of backtracks to do before return 0.0
    Returns
    -------
    the maximum step length coefficient for which constraints_satisfied evaluates to True
    采用了指数式下降来搜寻最佳的步长 alpha
    也就是说从一开始用最大的步长 alpha(0)
    接下来每一个学习率都在上一个备选值的基础上乘以缩减因子 coefficient= 0.5
    """

    step_len = max_step_len / line_search_coefficient

    for i in range(max_iter):
        step_len *= line_search_coefficient  # max_len, max_len*coefficient, max_len*coefficient^2, ...

        if constraints_satisfied(step_len * search_dir, step_len):
            return step_len

    return torch.tensor(0.0, dtype=torch.float32)
