# -*- coding: utf-8 -*-
"""
牛-拉法直角坐标系潮流计算

@禹亦凯

2022/10/13
"""
# TODO: 最好要再加一个可以自动载入数据，生成节点导纳矩阵的函数
# TODO: 暂时只测试了一个只有两个节点的网络，节点比较多的网络还有待测试
# TODO: 做面向对象重构

import numpy as np

"""节点导纳矩阵"""
# TODO: 可能要加一个生成节点导纳矩阵的函数，把那几个生成函数都放到其他文件里面
Y1 = np.array([[2 - 4.j, -2. + 4.j],
              [-2 + 4.j, 2 - 4.j]])

# 输入Y是可以只输入上三角矩阵
_ = np.triu(Y1, k=1)
Y = Y1 + _.T


"""给定PQ节点PV节点的已知量"""
Ps = np.array([-0.2])  # n-1
Qs = np.array([-0.1])  # m-1
U1 = np.array([])  # n-m,可以直接输入复数形式的
Us = np.array([np.abs(U1[i]) for i in range(len(U1))])

"""初始值"""
e = np.array([1, 1])  # n
f = np.array([0, 0])  # n

G = np.real(Y)
B = np.imag(Y)

"""这个n是实际的节点数n-1"""
n = len(Ps)
m = len(Qs)

"""初始化"""
delta_P = np.zeros(n)
delta_Q = np.zeros(m)
delta_U2 = np.zeros(n - m)

"""精度"""
EPSILON = 0.0001


def g_delta(nn, mm):
    """生成delta_P, delta_Q, delta_U2"""
    for i in range(nn):
        delta_P[i] = Ps[i] - sum(
            [e[i] * (G[i, j] * e[j] - B[i, j] * f[j]) + f[i] * (G[i, j] * f[j] + B[i, j] * e[j])
             for j in range(nn + 1)])

    for i in range(mm):
        # DONE 这里有问题
        delta_Q[i] = Qs[i] - sum(
            [f[i] * (G[i, j] * e[j] - B[i, j] * f[j]) - e[i] * (G[i, j] * f[j] + B[i, j] * e[j])
             for j in range(mm + 1)])

    for i in range(nn - mm):
        delta_U2[i] = Us[i] ** 2 - (e[i] ** 2 + f[i] ** 2)
    try:
        delta_PQU2 = np.vstack((delta_P.T, delta_Q.T, delta_U2.T))
    except:
        delta_PQU2 = np.vstack((delta_P.T, delta_Q.T))
    # for i in range(mm):
    #     delta_PQU2 = np.vstack((delta_PQU2, delta_P[i], delta_Q[i]))
    #
    # for i in range(mm, nn):
    #     delta_PQU2 = np.vstack((delta_PQU2, delta_P[i], delta_U2[i]))

    return delta_PQU2


def jacobi(nn, mm):
    """生成雅克比矩阵"""
    H = np.zeros((nn, nn))
    N = np.zeros((nn, nn))
    J = np.zeros((mm, nn))
    L = np.zeros((mm, nn))
    R = np.zeros((nn - mm, nn))
    S = np.zeros((nn - mm, nn))

    # DONE: 用一堆迭代分别求H, N, J, L, R, S
    for i in range(nn):
        for j in range(nn):
            if i != j:
                H[i, j] = B[i, j] * e[i] - G[i, j] * f[i]
            else:
                H[i, i] = B[i, i] * e[i] - G[i, i] * f[i] - \
                          sum([G[i, jj] * f[jj] + B[i, jj] * e[jj] for jj in range(nn + 1)])

    for i in range(nn):
        for j in range(nn):
            if i != j:
                N[i, j] = -G[i, j] * e[i] - B[i, j] * f[i]
            else:
                N[i, i] = -G[i, j] * e[i] - B[i, i] * f[i] - \
                          sum([G[i, jj] * e[jj] - B[i, j] * f[jj] for jj in range(nn + 1)])

    for i in range(mm):
        for j in range(nn):
            if i != j:
                J[i, j] = G[i, j] * e[i] + B[i, j] * f[i]
            else:
                J[i, j] = G[i, j] * e[i] + B[i, j] * f[i] - \
                          sum([G[i, jj] * e[jj] - B[i, j] * f[jj] for jj in range(nn + 1)])

    for i in range(mm):
        for j in range(nn):
            if i != j:
                L[i, j] = B[i, j] * e[i] - G[i, j] * f[i]
            else:
                L[i, i] = B[i, j] * e[i] - G[i, j] * f[i] + \
                          sum([G[i, jj] * f[jj] + B[i, jj] * e[jj] for jj in range(nn + 1)])

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
    i = 0
    while True:
        """main loop"""
        delta_PQU2 = g_delta(n, m)
        # DONE 雅克比矩阵这里有bug,求出来是奇异矩阵
        J = jacobi(n, m)
        inv_J = np.linalg.inv(J)
        delta_fe = np.dot(inv_J, delta_PQU2)
        # _ = f[:n].copy()
        # DONE 这里有一点问题，其他似乎是已经没问题了
        # fe = np.vstack((f, e))
        # fe[:, :n] = fe[:, :n] - delta_fe
        z = np.zeros((2, 1))
        delta_fe = np.c_[delta_fe, z]
        f = f - np.reshape(delta_fe[:n], np.shape(f))
        # _ = e[:n].copy()
        e = e - np.reshape(delta_fe[n:], np.shape(f))
        # e[:n] = _
        # [f[:n], e[:n]] = [f[:n], e[:n]] - delta_fe
        # error.append(max(delta_fe))
        error = np.append(error, delta_fe.max())
        print(f'第{i + 1}次迭代')
        print('error:', error[i])
        print(f'f:{f}, e:{e}')
        if error[i] <= EPSILON:
            break
        i += 1

