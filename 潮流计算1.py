import numpy as np


class PowerFlowCalculator:
    """潮流计算的类"""

    def __init__(self, Y, Ps, Qs, Us, f, e, epsilon=0.0001):
        """
        初始化
        以下除了epsilon都必须是ndarray类型
        Y:节点导纳矩阵（可以只写上三角矩阵）
        Ps:有功的给定量
        Qs:无功给定量
        Us:电压给定量
        f:节点电压虚部
        e:节点电压实部
        epsilon:精度（不写默认0.0001）
        """
        self.Ps = Ps
        self.Qs = Qs
        self.Us = np.array([np.abs(Us[i]) for i in range(len(Us))])
        self.Y = Y + np.triu(Y, k=1).T
        self.f = f
        self.e = e
        self.EPSILON = epsilon
        self.G = np.real(Y)
        self.B = np.imag(Y)
        self.n = len(self.Ps)
        self.m = len(self.Qs)
        self.delta_P = np.zeros(self.n)
        self.delta_Q = np.zeros(self.m)
        self.delta_U2 = np.zeros(self.n - self.m)

    def g_delta(self, nn, mm):
        """生成delta_P, delta_Q, delta_U2"""
        for i in range(nn):
            self.delta_P[i] = self.Ps[i] - sum(
                [self.e[i] * (self.G[i, j] * self.e[j] - self.B[i, j] * self.f[j]) + self.f[i] * (
                        self.G[i, j] * self.f[j] + self.B[i, j] * self.e[j])
                 for j in range(nn + 1)])

        for i in range(mm):
            # DONE 这里有问题
            self.delta_Q[i] = self.Qs[i] - sum(
                [self.f[i] * (self.G[i, j] * self.e[j] - self.B[i, j] * self.f[j]) - self.e[i] * (
                        self.G[i, j] * self.f[j] + self.B[i, j] * self.e[j])
                 for j in range(mm + 1)])

        for i in range(nn - mm):
            self.delta_U2[i] = self.Us[i] ** 2 - (self.e[i] ** 2 + self.f[i] ** 2)
        try:
            self.delta_PQU2 = np.vstack((self.delta_P.reshape(-1, 1), self.delta_Q.reshape(-1, 1), self.delta_U2.reshape(-1, 1)))
        except:
            self.delta_PQU2 = np.vstack((self.delta_P.reshape(-1, 1), self.delta_Q.reshape(-1, 1)))
        # for i in range(mm):
        #     delta_PQU2 = np.vstack((delta_PQU2, delta_P[i], delta_Q[i]))
        #
        # for i in range(mm, nn):
        #     delta_PQU2 = np.vstack((delta_PQU2, delta_P[i], delta_U2[i]))

        return self.delta_PQU2

    def jacobi(self, nn, mm):
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
                    H[i, j] = self.B[i, j] * self.e[i] - self.G[i, j] * self.f[i]
                else:
                    H[i, i] = self.B[i, i] * self.e[i] - self.G[i, i] * self.f[i] - \
                              sum([self.G[i, jj] * self.f[jj] + self.B[i, jj] * self.e[jj] for jj in range(nn + 1)])

        for i in range(nn):
            for j in range(nn):
                if i != j:
                    N[i, j] = -self.G[i, j] * self.e[i] - self.B[i, j] * self.f[i]
                else:
                    N[i, i] = -self.G[i, j] * self.e[i] - self.B[i, i] * self.f[i] - \
                              sum([self.G[i, jj] * self.e[jj] - self.B[i, j] * self.f[jj] for jj in range(nn + 1)])

        for i in range(mm):
            for j in range(nn):
                if i != j:
                    J[i, j] = self.G[i, j] * self.e[i] + self.B[i, j] * self.f[i]
                else:
                    J[i, j] = self.G[i, j] * self.e[i] + self.B[i, j] * self.f[i] - \
                              sum([self.G[i, jj] * self.e[jj] - self.B[i, j] * self.f[jj] for jj in range(nn + 1)])

        for i in range(mm):
            for j in range(nn):
                if i != j:
                    L[i, j] = self.B[i, j] * self.e[i] - self.G[i, j] * self.f[i]
                else:
                    L[i, i] = self.B[i, j] * self.e[i] - self.G[i, j] * self.f[i] + \
                              sum([self.G[i, jj] * self.f[jj] + self.B[i, jj] * self.e[jj] for jj in range(nn + 1)])

        for i in range(nn - mm):
            for j in range(nn):
                if i != j:
                    R[i, j] = 0
                else:
                    R[i, i] = -2 * self.f[i]

        for i in range(nn - mm):
            for j in range(nn):
                if i != j:
                    S[i, j] = 0
                else:
                    S[i, j] = -2 * self.e[i]

        HN = np.hstack((H, N))
        JL = np.hstack((J, L))
        RS = np.hstack((R, S))
        Ja = np.vstack((HN, JL, RS))
        return Ja

    def calculator(self):
        """主计算函数"""
        error = np.array([])
        i = 0
        while True:
            """main loop"""
            self.delta_PQU2 = self.g_delta(self.n, self.m)
            # DONE 雅克比矩阵这里有bug,求出来是奇异矩阵
            J = self.jacobi(self.n, self.m)
            inv_J = np.linalg.inv(J)
            delta_fe = np.dot(inv_J, self.delta_PQU2)
            # _ = f[:n].copy()
            # DONE 这里有一点问题，其他似乎是已经没问题了
            # fe = np.vstack((f, e))
            # fe[:, :n] = fe[:, :n] - delta_fe
            z = np.zeros((self.m + self.n + 1, 1))
            delta_fe = np.c_[delta_fe, z]
            self.f = self.f - np.reshape(delta_fe[:self.m], np.shape(self.f))
            # _ = e[:n].copy()
            self.e = self.e - np.reshape(delta_fe[self.m:], np.shape(self.f))
            # e[:n] = _
            # [f[:n], e[:n]] = [f[:n], e[:n]] - delta_fe
            # error.append(max(delta_fe))
            error = np.append(error, delta_fe.max())
            print(f'第{i + 1}次迭代')
            print('error:', error[i])
            print(f'f:{self.f}, e:{self.e}')
            if error[i] <= self.EPSILON:
                break
            i += 1
