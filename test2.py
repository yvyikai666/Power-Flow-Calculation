# 5个节点
import numpy as np
import 潮流计算1 as pfc

Y = np.array([
    [-33.3333j, 31.7460j, 0, 0, 0],
    [0, 1.5846 - 35.7379j, -0.8299 + 3.1120j, -0.7547 + 2.6415j, 0],
    [0, 0, 1.4539 - 66.9808j, -0.6240 + 3.9002j, 63.4921j],
    [0, 0, 0, 1.3787 - 6.2917j, 0],
    [0, 0, 0, 0, -66.6667j]])

S1 = np.array([-2, -3.7 - 1.3j, -2 - 1j, -1.6 - 0.8j, -1])
S2 = np.array([-2, -1, -1, -1, 5 + 1.05j])
U = np.array([1.05 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 1.05 + 0j])


Ps = np.array([-2, -3.7, -2, -1.6])
Qs = np.array([-1.3, -1, -0.8])
Us = np.array([1.05])

f = np.array([0, 0, 0, 0, 0])
e = np.array([1.05, 1, 1, 1, 1.05])


test2 = pfc.PowerFlowCalculator(Y, Ps, Qs, Us, f, e)
test2.calculator()