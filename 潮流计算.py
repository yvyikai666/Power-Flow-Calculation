# -*- coding: utf-8 -*-
"""
牛-拉法直角坐标系潮流计算

@禹亦凯

2022/10/13
"""

import numpy as np

"""节点导纳矩阵"""
# TODO: 可能要加一个生成节点导纳矩阵的函数，把那几个生成函数都放到其他文件里面
Y = np.array([])

"""给定PQ节点PV节点的已知量"""
Ps = np.array([])  # n-1
Qs = np.array([])  # m-1
Us = np.array([])  # n-m

"""初始值"""
e = np.array([])  # n
f = np.array([])  # n

G = np.real(Y)
B = np.imag(Y)

"""这个n是实际的n-1"""
n = len(Ps)
m = len(Qs)

"""初始化"""
delta_P = np.zeros((1, n))
delta_Q = np.zeros((1, m))
delta_U2 = np.zeros((1, n - m))

epison = 0.0001


def g_delta(nn, mm):
    """生成delta_P, delta_Q, delta_U2"""
    for i in range(nn + 1):
        delta_P[i] = Ps[i] - sum(
            [e[i] * (G[i, j] * e[j] - B[i, j] * f[j]) + f[i] * (G[i, j] * f[j] + B[i, j] * e[j])
             for j in range(nn + 1)])

    for i in range(mm + 1):
        delta_Q[i] = Qs[i] - sum(
            [f[i] * (G[i, j] * e[j] - B[i, j] * f[j]) + e[i] * (G[i, j] * f[j] + B[i, j] * e[j])
             for j in range(mm + 1)])

    for i in range(nn - mm):
        delta_U2[i] = Us[i] ** 2 - (e[i] ** 2 + f[i] ** 2)

    delta_PQU2 = np.array([])
    for i in range(mm):
        delta_PQU2 = np.vstack(delta_PQU2, delta_P[i], delta_Q[i])

    for i in range(mm, nn):
        delta_PQU2 = np.vstack(delta_PQU2, delta_P[i], delta_U2[i])

    return delta_PQU2


def jacobi(nn, mm):
    """生成雅克比矩阵"""
    H = np.zeros((nn, nn))
    N = np.zeros((nn, nn))
    J = np.zeros((mm, nn))
    L = np.zeros((mm, nn))
    R = np.zeros((nn - mm, nn))
    S = np.zeros((nn - mm, nn))

    # TODO: 用一堆迭代分别求H, N, J, L, R, S
    for i in range(nn):
        for j in range(nn):
            if i != j:
                H[i, j] = B[i, j] * e[i] - G[i, j] * f[i]
            else:
                H[i, i] = B[i, i] * e[i] - G[i, i] * f[i] - \
                          sum([G[i, jj] * f[jj] + B[i, jj] * e[jj] for jj in range(nn)])

    for i in range(nn):
        for j in range(nn):
            if i != j:
                N[i, j] = -G[i, j] * e[i] - B[i, j] * f[i]
            else:
                N[i, i] = -G[i, j] * e[i] - B[i, i] * f[i] - \
                          sum([G[i, jj] * e[jj] - B[i, j] * f[jj] for jj in range(nn)])

    for i in range(mm):
        for j in range(nn):
            if i != j:
                J[i, j] = G[i, j] * e[i] + B[i, j] * f[i]
            else:
                J[i, j] = G[i, j] * e[i] + B[i, j] * f[i] - \
                          sum([G[i, jj] * e[jj] - B[i, j] * f[jj] for jj in range(nn)])

    for i in range(mm):
        for j in range(nn):
            if i != j:
                L[i, j] = B[i, j] * e[i] - G[i, j] * f[i]
            else:
                L[i, i] = B[i, j] * e[i] - G[i, j] * f[i] + \
                          sum([G[i, jj] * f[jj] + B[i, jj] * e[jj] for jj in range(nn)])

    for i in range(nn - mm):
        for j in (nn):
            if i != j:
                R[i, j] = 0
            else:
                R[i, i] = -2 * f[i]

    for i in range(nn - mm):
        for j in range(nn):
            if i != j:
                S[i, j] = 0
            else:
                S[i, j] = -2 * e[i]

    HN = np.hstack((H, N))
    JL = np.hstack((J, L))
    RS = np.hstack((R, S))
    Ja = np.vstack((HN, JL, RS))
    return Ja


if __name__ == '__main__':
    error = np.array([])
    while True:
        """main loop"""
        i = 0
        delta_PQU2 = g_delta(n, m)
        J = jacobi(n, m)
        inv_J = np.inv(J)
        delta_fe = np.dot(inv_J, delta_PQU2)
        f += delta_fe[::2]
        e += delta_fe[1::2]
        # error.append(max(delta_fe))
        np.append(error, max(delta_fe))
        if error[i] <= epison:
            break
        i += 1
