"""
owner: Zou ying Cao
data: 2023-02-01
description:
"""
import numpy as np
from math import sin, asin, cos, radians, fabs, sqrt

EARTH_RADIUS = 6371  # 地球平均半径，6371km


def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):
    """
     用haversine公式计算球面两点间的距离
    """
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    d_lng = fabs(lng0 - lng1)
    d_lat = fabs(lat0 - lat1)
    h = hav(d_lat) + cos(lat0) * cos(lat1) * hav(d_lng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance


def gps_to_id(latitude, longitude):
    i = (latitude - 31.22) / 0.004
    if i < 0 or i > 9:
        i = np.random.randint(0, 10)
    j = (longitude - 121.45) / 0.004
    if j < 0 or j > 9:
        j = np.random.randint(0, 10)
    return int(j + 10 * i)


def cal_reset_duration(cur_lat, cur_long, re_lat, re_long):
    dis = get_distance_hav(cur_lat, cur_long, re_lat, re_long)
    # 时速设置和路径规划中的时速一样，也是将路径计算出的时间然后转化成时间步
    # 因为 位置reset 的时间要求可能不像路径规划 要求那么高
    return dis / 15 * 12


def generate_weights(count=1, n=3, m=1):
    all_weights = []

    target = np.random.dirichlet(np.ones(n), 1)[0]
    prev_t = target
    for _ in range(count // m):  # //表示整数除法
        target = np.random.dirichlet(np.ones(n), 1)[0]
        if m == 1:
            all_weights.append(target)
        else:
            for i in range(m):
                i_w = target * (i + 1) / float(m) + prev_t * \
                    (m - i - 1) / float(m)
                all_weights.append(i_w)
        prev_t = target + 0.

    return all_weights
