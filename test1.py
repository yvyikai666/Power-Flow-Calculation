import 潮流计算1 as pfc
import numpy as np
Y = np.array([[2 - 4.j, -2. + 4.j],
              [-2 + 4.j, 2 - 4.j]])


Ps = np.array([-0.2])  # n-1
Qs = np.array([-0.1])  # m-1
U1 = np.array([])  # n-m,可以直接输入复数形式的
Us = np.array([np.abs(U1[i]) for i in range(len(U1))])

"""初始值"""
e = np.array([1, 1])  # n
f = np.array([0, 0])  # n


test1 = pfc.PowerFlowCalculator(Y, Ps, Qs, Us, f, e)
test1.calculator()