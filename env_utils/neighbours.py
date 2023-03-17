"""
owner: Zou ying Cao
data: 2023-02-01
description:
"""
import os
import errno


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def ids_2dto1d(i, j, M, N):
    assert 0 <= i < M and 0 <= j < N
    index = i * N + j
    return int(index)


def ids_1dto2d(ids, M, N):
    i = int(ids / N)
    j = ids - N * i
    return int(i), int(j)


def get_neighbor_list(i, j, M, N, n, nodes):
    neighbor_list = [None] * n  # 边缘节点的不存在邻域节点设为None
    if n == 6:
        # hexagonal 六边形
        #   5 4
        # 0 x 3
        #   1 2
        if j % 2 == 0:
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i - 1, j, M, N)]
            if j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i, j + 1, M, N)]
            if i + 1 < M and j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i + 1, j + 1, M, N)]
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i + 1, j, M, N)]
            if i + 1 < M and j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i + 1, j - 1, M, N)]
            if j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i, j - 1, M, N)]
        elif j % 2 == 1:
            if i - 1 >= 0:
                neighbor_list[0] = nodes[ids_2dto1d(i - 1, j, M, N)]
            if i - 1 >= 0 and j + 1 < N:
                neighbor_list[1] = nodes[ids_2dto1d(i - 1, j + 1, M, N)]
            if j + 1 < N:
                neighbor_list[2] = nodes[ids_2dto1d(i, j + 1, M, N)]
            if i + 1 < M:
                neighbor_list[3] = nodes[ids_2dto1d(i + 1, j, M, N)]
            if j - 1 >= 0:
                neighbor_list[4] = nodes[ids_2dto1d(i, j - 1, M, N)]
            if i - 1 >= 0 and j - 1 >= 0:
                neighbor_list[5] = nodes[ids_2dto1d(i - 1, j - 1, M, N)]
    elif n == 4:
        # square 四邻域
        #   3
        # 0 x 2
        #   1
        if i - 1 >= 0:
            neighbor_list[0] = nodes[ids_2dto1d(i - 1, j, M, N)]
        if j + 1 < N:
            neighbor_list[1] = nodes[ids_2dto1d(i, j + 1, M, N)]
        if i + 1 < M:
            neighbor_list[2] = nodes[ids_2dto1d(i + 1, j, M, N)]
        if j - 1 >= 0:
            neighbor_list[3] = nodes[ids_2dto1d(i, j - 1, M, N)]
    elif n == 8:
        # 八邻域
        # 7 6 5
        # 0 x 4
        # 1 2 3
        if i - 1 >= 0:
            neighbor_list[0] = nodes[ids_2dto1d(i - 1, j, M, N)]
        if i - 1 >= 0 and j + 1 < N:
            neighbor_list[1] = nodes[ids_2dto1d(i - 1, j + 1, M, N)]
        if j + 1 < N:
            neighbor_list[2] = nodes[ids_2dto1d(i, j + 1, M, N)]
        if i + 1 < M and j + 1 < N:
            neighbor_list[3] = nodes[ids_2dto1d(i + 1, j + 1, M, N)]
        if i + 1 < M:
            neighbor_list[4] = nodes[ids_2dto1d(i + 1, j, M, N)]
        if i + 1 < M and j - 1 >= 0:
            neighbor_list[5] = nodes[ids_2dto1d(i + 1, j - 1, M, N)]
        if j - 1 >= 0:
            neighbor_list[6] = nodes[ids_2dto1d(i, j - 1, M, N)]
        if i - 1 >= 0 and j - 1 >= 0:
            neighbor_list[7] = nodes[ids_2dto1d(i - 1, j - 1, M, N)]

    return neighbor_list


def get_neighbor_index(i, j):  # 八邻域index
    neighbor_matrix_ids = [[i - 1, j],
                           [i - 1, j + 1],
                           [i, j + 1],
                           [i + 1, j + 1],
                           [i + 1, j],
                           [i + 1, j - 1],
                           [i, j - 1],
                           [i - 1, j - 1]
                           ]
    return neighbor_matrix_ids


def get_layers_neighbors(i, j, l_max, M, N):
    """
    get neighbors of node layer by layer, todo BFS.
    i, j: center node location
    L_max: max number of layers
    layers_neighbors: layers_neighbors[0] first layer neighbor: 6 nodes: can arrived in 1 time step.
    layers_neighbors[1]: 2nd layer nodes id
    M, N: matrix rows and columns.
    """
    assert l_max >= 1
    layers_neighbors = []
    temp_layer1_neighbor = get_neighbor_index(i, j)  # [[1,1], [0, 1], ...], 默认8邻域(1层)
    temp = []
    node_id_neighbors = []
    for item in temp_layer1_neighbor:  # 删选掉在地图之外的邻域节点序号
        x, y = item
        if 0 <= x <= M - 1 and 0 <= y <= N - 1:
            temp.append(item)
            node_id_neighbors.append(ids_2dto1d(x, y, M, N))
    layers_neighbors.append(temp)  # 2维序号
    layers_neighbors_set = set(node_id_neighbors)  # 一维序号集合

    curr_node_id = ids_2dto1d(i, j, M, N)
    layers_neighbors_set.add(curr_node_id)

    t = 1
    while t < l_max:
        layer_neighbor_temp = []
        for item in layers_neighbors[-1]:  # 第t层邻域节点2维序号
            x, y = item
            if 0 <= x <= M - 1 and 0 <= y <= N - 1:
                layer_neighbor_temp += get_neighbor_index(x, y)

        layer_neighbor = []  # remove previous layer neighbors
        for item in layer_neighbor_temp:
            x, y = item
            if 0 <= x <= M - 1 and 0 <= y <= N - 1:
                node_id = ids_2dto1d(x, y, M, N)
                if node_id not in layers_neighbors_set:
                    layer_neighbor.append(item)
                    layers_neighbors_set.add(node_id)
        layers_neighbors.append(layer_neighbor)
        t += 1

    return layers_neighbors
