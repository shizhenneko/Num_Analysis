import numpy as np
from scipy.misc import derivative
def f(x):
    return np.cos(x)
#对f(x)进行数值求导
def f_derivative(x):
    return derivative(f, x, dx=1e-6)

# 给出原函数列表y0和导数列表y1，给出插值点x0和x1，返回插值函数H(x)
# x0: 给定函数值的节点, y0: 对应的函数值
# x1: 给定导数值的节点, y1: 对应的导数值
def Hermite(x0: np.array, y0: np.array, y1: np.array):
    n = len(x0)  # 函数值节点数
    r = len(y1)  # 导数值节点数
    
    # Lagrange基函数(给定插值点数num = n或num = r)决定哪个基函数
    def lagrange_base(x_val,i,num):
        result = 1.0
        for j in range(num):
            if j != i:
                result *= (x_val - x0[j]) / (x0[i] - x0[j])
        return result
    
    # Lagrange基函数的导数
    def lagrange_base_derivative( i , num):
        result = 0
        for j in range(num):
            if j!= i:
                result += 1/ (x0[i] - x0[j])
        return result
    
    # 函数值的基函数 α_i(x)
    def alpha_basis(x_val, i):
        b = 1 - (x_val - x0[i])*lagrange_base_derivative( i, r) - (x_val - x0[i])*lagrange_base_derivative( i, n)
        return b*lagrange_base(x_val, i, r)*lagrange_base(x_val, i, n)

    
    # 导数值的基函数 β_i(x)
    def beta_basis(x_val, i):
        return (x_val - x0[i])*lagrange_base(x_val,i,n)*lagrange_base(x_val,i,r)
    
    # Hermite插值函数
    def hermite_interpolation(x_val):
        """计算Hermite插值多项式在x_val处的值"""
        result = 0.0
        
        # 添加函数值项
        for i in range(n):
            result += y0[i] * alpha_basis(x_val, i)
        # 添加导数值项
        for i in range(r):
            # x1中的节点在all_nodes中的索引为n+i
            result += y1[i] * beta_basis(x_val, i)
        return result
    return hermite_interpolation


# 测试示例
if __name__ == "__main__":
    print("=" * 60)
    print("测试1: 标准Hermite插值（所有节点都有导数信息）")
    print("=" * 60)
    
    x = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    y = np.array([f(xi) for xi in x])
    y_prime = np.array([f_derivative(xi) for xi in x])
    
    H = Hermite(x, y, y_prime)
    
    test_points = [np.pi/6, np.pi/3, 2*np.pi/3]
    for test_x in test_points:
        H_val = H(test_x)
        f_val = f(test_x)
        error = abs(H_val - f_val)
        print(f"x = {test_x:.4f}: H(x) = {H_val:.8f}, f(x) = {f_val:.8f}, 误差 = {error:.2e}")
    
    print("\n" + "=" * 60)
    print("测试2: 部分节点有导数信息（前3个节点）")
    print("=" * 60)
    
    r = 3  # 前3个节点有导数信息
    y_prime_partial = y_prime[:r]
    
    H2 = Hermite(x, y, y_prime_partial)
    
    for test_x in test_points:
        H_val = H2(test_x)
        f_val = f(test_x)
        error = abs(H_val - f_val)
        print(f"x = {test_x:.4f}: H(x) = {H_val:.8f}, f(x) = {f_val:.8f}, 误差 = {error:.2e}")