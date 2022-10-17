# 牛顿法潮流计算 直角坐标
import numpy as np

# 输入数据
'''
节点导纳矩阵 Y , 只输入上三角即可
节点功率 S , 包括PQ节点的 S1，PV节点的 S2
平衡节点值设为-2，-1表示另一种节点，S2的实部表示P，虚部表示V
节点电压 U 初值
'''
Y = np.array([
    [-33.3333j, 31.7460j, 0, 0, 0],
    [0, 1.5846 - 35.7379j, -0.8299 + 3.1120j, -0.7547 + 2.6415j, 0],
    [0, 0, 1.4539 - 66.9808j, -0.6240 + 3.9002j, 63.4921j],
    [0, 0, 0, 1.3787 - 6.2917j, 0],
    [0, 0, 0, 0, -66.6667j]
])
S1 = np.array([-2, -3.7 - 1.3j, -2 - 1j, -1.6 - 0.8j, -1])
S2 = np.array([-2, -1, -1, -1, 5 + 1.05j])
U = np.array([1.05 + 0j, 1 + 0j, 1 + 0j, 1 + 0j, 1.05 + 0j])
# 数据预处理，初始化
'''
1. 将节点导纳矩阵补全
2. 设定迭代次数 k
3. 由导纳矩阵分割为矩阵 G 和 B
4. 由电压值生成实部和虚部的 E 和 F
5. 生成排除平衡节点的电压值 ef
'''
for i in range(len(Y)):
    for j in range(len(Y[i])):
        if i > j:
            Y[i][j] = Y[j][i]
k = 0
G = []
B = []
for x in Y:
    temp1 = []
    temp2 = []
    for y in x:
        temp1.append(y.real)
        temp2.append(y.imag)
    G.append(temp1)
    B.append(temp2)
G = np.asarray(G)
B = np.asarray(B)
E = []
F = []
for z in U:
    E.append(z.real)
    F.append(z.imag)
E = np.asarray(E)
F = np.asarray(F)
# 顺便 产生一个在 n 中标记 PV 节点的数
mark_PV = []
temp3 = 0
ef = []
for i in range(len(U)):
    if (S1[i] == -1):
        mark_PV.append(temp3)
    temp3 = temp3 + 1
    if (S1[i] != -2):
        ef.append(U[i].real)
        ef.append(U[i].imag)
ef = np.asarray(ef)
# 计算功率不平衡量 和电压不平衡量 ERO
'''参数说明
PQ节点的P和Q，即S1	全局变量
PV节点的P和V，即S2	全局变量
U_real	迭代电压的实部，含平衡节点
U_imag	迭代电压的虚部
G和B 	导纳的实部值 和 导纳的虚部值，全局变量
'''


def cal_ERO(U_real, U_imag):
    ERO1 = []
    ERO_k1 = []  # 用于判断收敛情况
    for i in range(len(U_real)):
        if (S1[i] != -2 and S1[i] != -1):  # PQ 平衡节点不需要计算，需要排除
            temp1 = 0
            temp2 = 0
            for j in range(len(U_real)):
                temp1 = temp1 + G[i][j] * U_real[j] - B[i][j] * U_imag[j]
                temp2 = temp2 + G[i][j] * U_imag[j] + B[i][j] * U_real[j]
            ERO1.append(round(S1[i].real - U_real[i] * temp1 - U_imag[i] * temp2, 5))
            ERO1.append(round(S1[i].imag - U_imag[i] * temp1 + U_real[i] * temp2, 5))
            ERO_k1.append(round(S1[i].real - U_real[i] * temp1 - U_imag[i] * temp2, 5))
            ERO_k1.append(round(S1[i].imag - U_imag[i] * temp1 + U_real[i] * temp2, 5))
        if (S2[i] != -2 and S2[i] != -1):  # PV
            temp1 = 0
            temp2 = 0
            for j in range(len(U_real)):
                temp1 = temp1 + G[i][j] * U_real[j] - B[i][j] * U_imag[j]
                temp2 = temp2 + G[i][j] * U_imag[j] + B[i][j] * U_real[j]
            ERO1.append(round(S2[i].real - U_real[i] * temp1 - U_imag[i] * temp2, 5))
            ERO1.append(round(S2[i].imag ** 2 - U_real[i] ** 2 - U_imag[i] ** 2, 5))
            ERO_k1.append(round(S2[i].real - U_real[i] * temp1 - U_imag[i] * temp2, 5))
    ERO1 = np.array(ERO1)
    ERO_k1 = np.array(ERO_k1)
    return ERO1, ERO_k1


# 计算雅克比矩阵
def cal_Jacobi(U_real, U_imag):
    J1 = []
    for i in range(len(E)):
        if (S1[i] != -2):  # 平衡节点排除
            temp1 = []
            temp2 = []
            # 计算第i行
            for j in range(len(E)):
                if (S1[j] != -2):
                    if (i not in mark_PV):  # 判断PQ节点
                        if (i != j):
                            # J1的第2i行
                            temp1.append(round((-1) * G[i][j] * U_real[i] - B[i][j] * U_imag[i], 5))
                            temp1.append(round(B[i][j] * U_real[i] - G[i][j] * U_imag[i], 5))
                            # J1的第2i+1行
                            temp2.append(round(B[i][j] * U_real[i] - G[i][j] * U_imag[i], 5))
                            temp2.append(round(G[i][j] * U_real[i] + B[i][j] * U_imag[i], 5))
                        else:
                            # J1的第2i行
                            temp3 = 0
                            for m in range(len(G)):
                                temp3 = temp3 - G[i][m] * U_real[m] + B[i][m] * U_imag[m]
                            temp1.append(round(temp3 - G[i][i] * U_real[i] - B[i][i] * U_imag[i], 5))
                            temp3 = 0
                            for m in range(len(G)):
                                temp3 = temp3 - G[i][m] * U_imag[m] - B[i][m] * U_real[m]
                            temp1.append(round(temp3 + B[i][i] * U_real[i] - G[i][i] * U_imag[i], 5))
                            # J1的第2i+1行
                            temp3 = 0
                            for m in range(len(G)):
                                temp3 = temp3 + G[i][m] * U_imag[m] + B[i][m] * U_real[m]
                            temp2.append(round(temp3 + B[i][i] * U_real[i] - G[i][i] * U_imag[i], 5))
                            temp3 = 0
                            for m in range(len(G)):
                                temp3 = temp3 - G[i][m] * U_real[m] + B[i][m] * U_imag[m]
                            temp2.append(round(temp3 + G[i][i] * U_real[i] + B[i][i] * U_imag[i], 5))
                    if (i in mark_PV):
                        if (i != j):
                            # J1的第2i行
                            temp1.append(round((-1) * G[i][j] * U_real[i] - B[i][j] * U_imag[i], 5))
                            temp1.append(round(B[i][j] * U_real[i] - G[i][j] * U_imag[i], 5))
                            # J1的第2i+1行
                            temp2.append(0)
                            temp2.append(0)
                        else:
                            # J1的第2i行
                            temp3 = 0
                            for m in range(len(G)):
                                temp3 = temp3 - G[i][m] * U_real[m] + B[i][m] * U_imag[m]
                            temp1.append(round(temp3 - G[i][i] * U_real[i] - B[i][i] * U_imag[i], 5))
                            temp3 = 0
                            for m in range(len(G)):
                                temp3 = temp3 - G[i][m] * U_imag[m] - B[i][m] * U_real[m]
                            temp1.append(round(temp3 + B[i][i] * U_real[i] - G[i][i] * U_imag[i], 5))
                            # J1的第2i+1行
                            temp2.append((-2) * U_real[i])
                            temp2.append((-2) * U_imag[i])
            J1.append(temp1)
            J1.append(temp2)
    J1 = np.array(J1)
    return J1


# 解修正方程 ERO = J·CO
# 并求得新的节点电压值 ef = ef - CO
# 求得新的 E 和 F
def CO_solve(J2, ERO2):
    global ef  # 声明这是全局变量
    global E
    global F
    CO1 = np.linalg.solve(J2, ERO2)
    # for i in CO1:
    # 	print(str(round(i,5)).center(10),end="")
    # print("")
    # 保留小数位数4
    for i in range(len(CO1)):
        CO1[i] = round(CO1[i], 8)
    # 修正节点电压值
    ef = ef - CO1
    # 更新矩阵 E 和F 的值
    p = 0
    for i in range(len(E)):
        if (S1[i] != -2):
            E[i] = ef[2 * p]
            F[i] = ef[2 * p + 1]
            p = p + 1


########### 主程序
print("历次迭代中的节点电压值变化:\n")
while True:
    # 求不平衡量
    ERO, ERO_k = cal_ERO(E, F)
    # 存储历次误差值
    if (k == 0):
        PQ_ERO = np.zeros((len(ERO_k),))
    PQ_ERO = np.vstack((PQ_ERO, ERO_k))
    # 求雅克比矩阵
    J = cal_Jacobi(E, F)
    CO_solve(J, ERO)
    k = k + 1
    # 打印历次节点电压值
    for i in ef:
        print(str('%.4f' % i).center(10), end="")
    print("")
    # 判断是否收敛
    if (max(list(map(abs, ERO_k))) <= 0.0000):
        flag = 1
        break
    # 或者说计算次数过多，也跳出循环
    if (k >= 8):
        flag = 0
        break

# 打印历次误差值
print("\n\n", "-" * 80, "\n历次迭代中的功率不平衡量:\n")
for s in PQ_ERO:
    for i in s:
        print(str('%.4f' % i).center(10), end="")
    print("")

# 迭代是否收敛
print("\n\n", "-" * 80, "\n迭代是否收敛:")
if (flag == 1):
    print("\t收敛")
else:
    print("\t不收敛")
